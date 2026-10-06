"""Pixel versions of the RSSM and JEPA families for the DMControl main grid.

Same recurrent / joint-embedding cores as the LunarLander models (wm/rssm.py,
wm/jepa.py), with a DreamerV3-style convolutional encoder and decoder at
64x64.  A `width` multiplier scales every channel and hidden size, giving the
plan's 0.5x and 2x model-size variants (plan Section 2.2, Pop-B).  Images
enter as raw 0-255 values in any dtype and are normalised to [-0.5, 0.5]
inside the encoder, so the numpy-in / numpy-out adapters work unchanged.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wm.rssm import RSSM, RSSMAdapter, symlog, mlp
from wm.jepa import JEPA, JEPAAdapter, vicreg_reg


class ConvEncoder(nn.Module):
    """(..., 64, 64, 3) raw pixels -> (..., embed)."""
    def __init__(self, embed, width=1.0):
        super().__init__()
        ch = [max(8, int(c * width)) for c in (32, 64, 128, 256)]
        layers, cin = [], 3
        for c in ch:
            layers += [nn.Conv2d(cin, c, 4, 2, 1), nn.GroupNorm(1, c), nn.SiLU()]
            cin = c
        self.net = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Linear(ch[-1] * 16, embed), nn.LayerNorm(embed), nn.SiLU())

    def forward(self, x):
        lead = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:]).float() / 255.0 - 0.5
        x = self.net(x.permute(0, 3, 1, 2))
        return self.out(x.flatten(1)).reshape(*lead, -1)


class ConvDecoder(nn.Module):
    """(..., feat) -> (..., 64, 64, 3) in normalised units [-0.5, 0.5]."""
    def __init__(self, feat, width=1.0):
        super().__init__()
        ch = [max(8, int(c * width)) for c in (256, 128, 64, 32)]
        self.c0 = ch[0]
        self.inp = nn.Linear(feat, ch[0] * 16)
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.ConvTranspose2d(a, b, 4, 2, 1), nn.GroupNorm(1, b), nn.SiLU()]
        layers += [nn.ConvTranspose2d(ch[-1], 3, 4, 2, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, f):
        lead = f.shape[:-1]
        x = self.inp(f.reshape(-1, f.shape[-1])).reshape(-1, self.c0, 4, 4)
        return self.net(x).permute(0, 2, 3, 1).reshape(*lead, 64, 64, 3)


class PixelRSSM(RSSM):
    def __init__(self, act_dim, width=1.0, embed=512):
        h, z, hid = int(512 * width), 32, int(512 * width)
        super().__init__(obs_dim=1, act_dim=act_dim, h_dim=h, z_dim=z, hidden=hid, embed=embed)
        self.enc = ConvEncoder(embed, width)
        self.dec = ConvDecoder(h + z, width)


def pixel_rssm_loss(model, obs, act, rew, done, free_nats=1.0, kl_balance=0.8):
    """DreamerV2/V3 recipe with an image likelihood summed over pixels."""
    h, z, priors, posts = model.observe(obs, act, sample=True)
    feat = torch.cat([h, z], -1)
    o_hat, r_hat, c_logit = model.heads(feat)
    target = obs.float() / 255.0 - 0.5
    recon = F.mse_loss(o_hat.float(), target, reduction="none").flatten(2).sum(-1).mean()
    r_loss = F.mse_loss(r_hat[:, 1:].float(), symlog(rew))
    c_loss = F.binary_cross_entropy_with_logits(c_logit[:, 1:].float(), (~done).float())
    pm = torch.stack([p[0] for p in priors], 1).float(); ps = torch.stack([p[1] for p in priors], 1).float()
    qm = torch.stack([q[0] for q in posts], 1).float(); qs = torch.stack([q[1] for q in posts], 1).float()
    kl = lambda m1, s1, m2, s2: (torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5).sum(-1)
    kl_val = kl_balance * torch.clamp(kl(qm.detach(), qs.detach(), pm, ps).mean(), min=free_nats) + \
        (1 - kl_balance) * torch.clamp(kl(qm, qs, pm.detach(), ps.detach()).mean(), min=free_nats)
    loss = recon + r_loss + c_loss + kl_val
    return loss, {"loss": loss.item(), "recon": recon.item(), "reward": r_loss.item(),
                  "kl": kl(qm, qs, pm, ps).mean().item()}


class PixelJEPA(JEPA):
    def __init__(self, act_dim, width=1.0, z_dim=192):
        hid = int(512 * width)
        super().__init__(obs_dim=1, act_dim=act_dim, z_dim=z_dim, hidden=hid)
        self.enc = ConvEncoder(z_dim, width)
        # the encoder ends in SiLU; project to an unconstrained latent for VICReg
        self.enc = nn.Sequential(self.enc, nn.Linear(z_dim, z_dim))


def pixel_jepa_loss(model, obs, act, rew, done, H=5, w_var=1.0, w_cov=0.04):
    e = model.enc(obs).float()
    B, T1, _ = e.shape
    pred_loss, n = 0.0, 0
    for s in range(0, T1 - H, H):
        z = e[:, s]
        for k in range(H):
            z = model.step(z, act[:, s + k]).float()
            pred_loss = pred_loss + F.mse_loss(z, e[:, s + k + 1]); n += 1
    pred_loss = pred_loss / n
    var_l, cov_l = vicreg_reg(e)
    ed = e[:, 1:].detach()
    r_loss = F.mse_loss(model.rew(ed).squeeze(-1).float(), symlog(rew))
    c_loss = F.binary_cross_entropy_with_logits(model.cont(ed).squeeze(-1).float(), (~done).float())
    loss = pred_loss + w_var * var_l + w_cov * cov_l + r_loss + c_loss
    return loss, {"loss": loss.item(), "pred": float(pred_loss), "var": var_l.item(),
                  "cov": cov_l.item(), "reward": r_loss.item()}


class PixelRSSMAdapter(RSSMAdapter):
    family = "rssm"


class PixelJEPAAdapter(JEPAAdapter):
    family = "jepa"


# ---------------------------------------------------------------------------
# JEPA with a 3-frame history (plan amendment A11)
# ---------------------------------------------------------------------------
class PixelJEPAHist(nn.Module):
    """LeWorldModel-style JEPA whose predictor sees the last K = 3 frame
    embeddings, as LeWM's does (history_size 3).  A single-frame encoding
    with a memoryless predictor cannot represent velocity from pixels (the
    first pixel JEPA's velocity probe had R^2 ~ 0.05), which would confound
    the family class with an architecture that cannot see motion.

    Latent state Z = concat(e_{t-2}, e_{t-1}, e_t).  step() predicts the next
    embedding as a residual on e_t and shifts the stack.  Reward and
    continuation read out from the detached stack, so velocity-dependent
    rewards (walker's speed term) are expressible."""
    K = 3

    def __init__(self, act_dim, width=1.0, z_dim=192):
        super().__init__()
        hid = int(512 * width)
        self.obs_dim, self.act_dim, self.z_dim = 1, act_dim, z_dim
        self.enc = nn.Sequential(ConvEncoder(z_dim, width), nn.Linear(z_dim, z_dim))
        self.pred = mlp(self.K * z_dim + act_dim, hid, z_dim)
        self.rew = mlp(self.K * z_dim, hid, 1)
        self.cont = mlp(self.K * z_dim, hid, 1)

    def step(self, Z, a):
        e_new = Z[:, -self.z_dim:] + self.pred(torch.cat([Z, a], -1))
        return torch.cat([Z[:, self.z_dim:], e_new], -1)

    def stack(self, e):
        """(B, T+1, z) embeddings -> (B, T+1, K*z) stacks; the first frames
        repeat e_0 (zero velocity) until the history fills."""
        idx = torch.arange(e.shape[1], device=e.device)
        parts = [e[:, torch.clamp(idx - (self.K - 1 - k), min=0)] for k in range(self.K)]
        return torch.cat(parts, -1)


def pixel_jepa_hist_loss(model, obs, act, rew, done, H=5, w_var=1.0, w_cov=0.04):
    e = model.enc(obs).float()                        # (B, T+1, z)
    Zs = model.stack(e)                               # (B, T+1, K*z)
    B, T1, zd = e.shape
    pred_loss, n = 0.0, 0
    for s in range(model.K - 1, T1 - H, H):           # start once the history is real
        Z = Zs[:, s]
        for k in range(H):
            Z = model.step(Z, act[:, s + k]).float()
            pred_loss = pred_loss + F.mse_loss(Z[:, -zd:], e[:, s + k + 1]); n += 1
    pred_loss = pred_loss / n
    var_l, cov_l = vicreg_reg(e)
    Zd = Zs[:, 1:].detach()                           # reward r_t arrives with frame t+1
    r_loss = F.mse_loss(model.rew(Zd).squeeze(-1).float(), symlog(rew))
    c_loss = F.binary_cross_entropy_with_logits(model.cont(Zd).squeeze(-1).float(), (~done).float())
    loss = pred_loss + w_var * var_l + w_cov * cov_l + r_loss + c_loss
    return loss, {"loss": loss.item(), "pred": float(pred_loss), "var": var_l.item(),
                  "cov": cov_l.item(), "reward": r_loss.item()}


class PixelJEPAHistAdapter(JEPAAdapter):
    """Markovian in the stacked-embedding state: filter_step shifts in the new
    frame's embedding, encode_sequence builds the stacks from a sequence, and a
    lone frame is encoded as a stationary history."""
    family = "jepa"

    def __init__(self, model, device="cpu"):
        super().__init__(model, device)
        self.latent_dim = model.K * model.z_dim

    @torch.no_grad()
    def encode(self, obs):
        e = self.m.enc(self._t(obs))
        return torch.cat([e] * self.m.K, -1).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def encode_sequence(self, obs, actions):
        e = self.m.enc(self._t(obs))
        return self.m.stack(e).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def filter_step(self, Z, a, obs):
        e = self.m.enc(self._t(obs))
        Z = self._t(Z)
        return torch.cat([Z[:, self.m.z_dim:], e], -1).cpu().numpy().astype(np.float64)
