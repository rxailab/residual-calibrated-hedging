"""Recurrent state-space model (DreamerV3-family RSSM) and its adapter.

Sized as in the LunarLander study being replicated (arXiv 2607.01736): a GRU
with a 256-d deterministic state and a 16-d Gaussian stochastic state.  The
latent the metric library sees is concat(h, z), 272-d.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wm.protocol import BaseAdapter


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class RSSM(nn.Module):
    def __init__(self, obs_dim=8, act_dim=2, h_dim=256, z_dim=16, hidden=256,
                 embed=128, min_std=0.1):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.h_dim, self.z_dim, self.min_std = h_dim, z_dim, min_std
        self.enc = mlp(obs_dim, hidden, embed)
        self.img_in = nn.Sequential(nn.Linear(z_dim + act_dim, hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, h_dim)
        self.prior_net = mlp(h_dim, hidden, 2 * z_dim, n=1)
        self.post_net = mlp(h_dim + embed, hidden, 2 * z_dim, n=1)
        feat = h_dim + z_dim
        self.dec = mlp(feat, hidden, obs_dim)
        self.rew = mlp(feat, hidden, 1)
        self.cont = mlp(feat, hidden, 1)

    def _dist(self, x):
        mu, s = x.chunk(2, -1)
        return mu, F.softplus(s) + self.min_std

    def initial(self, n, device):
        return torch.zeros(n, self.h_dim, device=device)

    def img_step(self, h, z, a, sample=False):
        h = self.gru(self.img_in(torch.cat([z, a], -1)), h)
        mu, std = self._dist(self.prior_net(h))
        z = mu + std * torch.randn_like(std) if sample else mu
        return h, z, (mu, std)

    def post(self, h, e, sample=True):
        mu, std = self._dist(self.post_net(torch.cat([h, e], -1)))
        z = mu + std * torch.randn_like(std) if sample else mu
        return z, (mu, std)

    def observe(self, obs, act, sample=True):
        """Posterior filtering.  obs (B,T+1,O), act (B,T,A).
        Returns h, z (B,T+1,.), prior and posterior params for t=1..T."""
        B, T1 = obs.shape[:2]          # works for vectors and images
        e = self.enc(obs)
        h = self.initial(B, obs.device)
        z, post0 = self.post(h, e[:, 0], sample)
        hs, zs, priors, posts = [h], [z], [], []
        for t in range(T1 - 1):
            h, _, pr = self.img_step(h, z, act[:, t])
            z, po = self.post(h, e[:, t + 1], sample)
            hs.append(h); zs.append(z); priors.append(pr); posts.append(po)
        return torch.stack(hs, 1), torch.stack(zs, 1), priors, posts

    def heads(self, feat):
        return self.dec(feat), self.rew(feat).squeeze(-1), self.cont(feat).squeeze(-1)


def rssm_loss(model, obs, act, rew, done, free_nats=1.0, kl_balance=0.8, kl_scale=1.0):
    """ELBO with KL balancing and free nats (DreamerV2/V3 recipe).
    Index convention: reward r_t and done_t arrive with observation t+1."""
    h, z, priors, posts = model.observe(obs, act, sample=True)
    feat = torch.cat([h, z], -1)
    o_hat, r_hat, c_logit = model.heads(feat)
    recon = F.mse_loss(o_hat, obs, reduction="none").sum(-1).mean()
    r_loss = F.mse_loss(r_hat[:, 1:], symlog(rew))
    c_loss = F.binary_cross_entropy_with_logits(c_logit[:, 1:], (~done).float())
    pm = torch.stack([p[0] for p in priors], 1); ps = torch.stack([p[1] for p in priors], 1)
    qm = torch.stack([q[0] for q in posts], 1); qs = torch.stack([q[1] for q in posts], 1)

    def kl(m1, s1, m2, s2):
        return (torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5).sum(-1)

    kl_lhs = kl(qm.detach(), qs.detach(), pm, ps)   # trains the prior
    kl_rhs = kl(qm, qs, pm.detach(), ps.detach())   # regularises the posterior
    kl_val = kl_balance * torch.clamp(kl_lhs.mean(), min=free_nats) + \
        (1 - kl_balance) * torch.clamp(kl_rhs.mean(), min=free_nats)
    loss = recon + r_loss + c_loss + kl_scale * kl_val
    return loss, {"loss": loss.item(), "recon": recon.item(), "reward": r_loss.item(),
                  "cont": c_loss.item(), "kl": kl(qm, qs, pm, ps).mean().item()}


class RSSMAdapter(BaseAdapter):
    """Exposes the RSSM through the study's interface.  Numpy in, numpy out;
    torch on `device` inside."""

    family = "dreamerv3_rssm"
    has_decoder = True
    has_native_reward = True
    is_stochastic = True

    def __init__(self, model: RSSM, device="cpu"):
        self.m = model.to(device).eval()
        self.dev = device
        self.latent_dim = model.h_dim + model.z_dim
        self.action_dim = model.act_dim
        self._probe = None

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

    def _split(self, zt):
        return zt[..., :self.m.h_dim], zt[..., self.m.h_dim:]

    # --- encoding -----------------------------------------------------------
    @torch.no_grad()
    def encode_sequence(self, obs, actions):
        h, z, _, _ = self.m.observe(self._t(obs), self._t(actions), sample=False)
        return torch.cat([h, z], -1).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def encode(self, obs):
        """Markovian fallback (no history): posterior from h = 0.  Only used
        where a single frame is all there is; closed-loop control uses
        filter_step so the latent carries the episode's history."""
        o = self._t(obs)
        h = self.m.initial(o.shape[0], self.dev)
        z, _ = self.m.post(h, self.m.enc(o), sample=False)
        return torch.cat([h, z], -1).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def filter_step(self, zt, a, obs):
        h, z = self._split(self._t(zt))
        h, _, _ = self.m.img_step(h, z, self._t(a))
        z, _ = self.m.post(h, self.m.enc(self._t(obs)), sample=False)
        return torch.cat([h, z], -1).cpu().numpy().astype(np.float64)

    # --- dynamics -----------------------------------------------------------
    @torch.no_grad()
    def step(self, zt, a):
        h, z = self._split(self._t(zt))
        h, z, _ = self.m.img_step(h, z, self._t(a), sample=False)
        return torch.cat([h, z], -1).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def rollout(self, z0, actions):
        acts = self._t(actions)
        h, z = self._split(self._t(z0))
        out = [torch.cat([h, z], -1)]
        for t in range(acts.shape[1]):
            h, z, _ = self.m.img_step(h, z, acts[:, t], sample=False)
            out.append(torch.cat([h, z], -1))
        return torch.stack(out, 1).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def sample(self, zt, a, n):
        h0, z0 = self._split(self._t(zt))
        outs = []
        for _ in range(n):
            h, z, _ = self.m.img_step(h0, z0, self._t(a), sample=True)
            outs.append(torch.cat([h, z], -1))
        return torch.stack(outs).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def reward(self, zt, a=None):
        return symexp(self.m.rew(self._t(zt)).squeeze(-1)).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def decode(self, zt):
        return self.m.dec(self._t(zt)).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def get_cost(self, z0, actions):
        """stable-worldmodel Costable.get_cost: negated predicted return over
        the horizon, each reward weighted by the predicted probability that the
        episode is still running (a crash ends reward accumulation)."""
        acts = self._t(actions)
        B = acts.shape[0]
        zt = self._t(z0)
        if zt.shape[0] == 1 and B > 1:
            zt = zt.expand(B, -1)
        h, z = self._split(zt)
        ret = torch.zeros(B, device=self.dev)
        alive = torch.ones(B, device=self.dev)
        for t in range(acts.shape[1]):
            h, z, _ = self.m.img_step(h, z, acts[:, t], sample=False)
            f = torch.cat([h, z], -1)
            ret = ret + alive * symexp(self.m.rew(f).squeeze(-1))
            alive = alive * torch.sigmoid(self.m.cont(f).squeeze(-1))
        return (-ret).cpu().numpy().astype(np.float64)

    # --- family-agnostic hooks used by the evaluator -------------------------
    def readout(self, zt):
        return self.decode(zt)

    @torch.no_grad()
    def cont_prob(self, zt):
        return torch.sigmoid(self.m.cont(self._t(zt)).squeeze(-1)).cpu().numpy()

    def torch_fns(self):
        m, hd = self.m, self.m.h_dim
        def step(x, a):
            h, z, _ = m.img_step(x[:hd][None], x[hd:][None], a[None], sample=False)
            return torch.cat([h, z], -1)[0]
        return step, (lambda x: m.dec(x[None])[0]), (lambda x: m.rew(x[None])[0, 0])
