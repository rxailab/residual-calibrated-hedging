"""Latent-predictive self-supervised world model (LeWorldModel-style JEPA).

The latent is shaped ONLY by self-supervised prediction plus an anti-collapse
regulariser; no reconstruction and no reward gradient reaches the encoder.
Reward and continuation heads read out from detached latents, the plan's
"reward head fitted on frozen latents" (Section 4).  This is the family class
in which the TwoRoom study found latent error anti-predictive, so running it
beside the RSSM on the same data is a direct test of hypothesis H2a.

Deviation from LeWM, recorded for the pre-registration: VICReg's variance and
covariance terms stand in for SIGReg as the anti-collapse regulariser.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wm.protocol import BaseAdapter
from wm.rssm import mlp, symlog, symexp


class JEPA(nn.Module):
    def __init__(self, obs_dim=8, act_dim=2, z_dim=64, hidden=256):
        super().__init__()
        self.obs_dim, self.act_dim, self.z_dim = obs_dim, act_dim, z_dim
        self.enc = mlp(obs_dim, hidden, z_dim)
        self.pred = mlp(z_dim + act_dim, hidden, z_dim)
        self.rew = mlp(z_dim, hidden, 1)
        self.cont = mlp(z_dim, hidden, 1)

    def step(self, z, a):
        return z + self.pred(torch.cat([z, a], -1))       # residual predictor


def vicreg_reg(e):
    e = e.reshape(-1, e.shape[-1])
    e = e - e.mean(0)
    std = torch.sqrt(e.var(0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    cov = (e.T @ e) / (e.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss, (off ** 2).sum() / e.shape[1]


def jepa_loss(model, obs, act, rew, done, H=5, w_var=1.0, w_cov=0.04):
    """Free-running H-step latent prediction, no stop-gradient (LeWM trains the
    encoder and predictor jointly and relies on the regulariser against
    collapse); heads trained on detached encoder latents."""
    e = model.enc(obs)                                     # (B, T+1, z)
    B, T1, _ = e.shape
    starts = torch.arange(0, T1 - H, H, device=obs.device)
    pred_loss = 0.0
    for s in starts:
        z = e[:, s]
        for k in range(H):
            z = model.step(z, act[:, s + k])
            pred_loss = pred_loss + F.mse_loss(z, e[:, s + k + 1])
    pred_loss = pred_loss / (len(starts) * H)
    var_l, cov_l = vicreg_reg(e)
    ed = e[:, 1:].detach()
    r_loss = F.mse_loss(model.rew(ed).squeeze(-1), symlog(rew))
    c_loss = F.binary_cross_entropy_with_logits(model.cont(ed).squeeze(-1), (~done).float())
    loss = pred_loss + w_var * var_l + w_cov * cov_l + r_loss + c_loss
    return loss, {"loss": loss.item(), "pred": float(pred_loss), "var": var_l.item(),
                  "cov": cov_l.item(), "reward": r_loss.item(), "cont": c_loss.item(),
                  "latent_std": float(e.reshape(-1, e.shape[-1]).std(0).mean())}


class JEPAAdapter(BaseAdapter):
    family = "jepa_ssl"
    has_decoder = False
    has_native_reward = False      # a readout head on frozen latents
    is_stochastic = False

    def __init__(self, model: JEPA, device="cpu"):
        self.m = model.to(device).eval(); self.dev = device
        self.latent_dim, self.action_dim = model.z_dim, model.act_dim
        self._probe = None

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

    @torch.no_grad()
    def encode(self, obs):
        return self.m.enc(self._t(obs)).cpu().numpy().astype(np.float64)

    def filter_step(self, z, a, obs):          # Markovian: the latent is the encoding
        return self.encode(obs)

    @torch.no_grad()
    def step(self, z, a):
        return self.m.step(self._t(z), self._t(a)).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def rollout(self, z0, actions):
        acts = self._t(actions); z = self._t(z0); out = [z]
        for t in range(acts.shape[1]):
            z = self.m.step(z, acts[:, t]); out.append(z)
        return torch.stack(out, 1).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def reward(self, z, a=None):
        return symexp(self.m.rew(self._t(z)).squeeze(-1)).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def cont_prob(self, z):
        return torch.sigmoid(self.m.cont(self._t(z)).squeeze(-1)).cpu().numpy()

    @torch.no_grad()
    def get_cost(self, z0, actions):
        acts = self._t(actions); B = acts.shape[0]; z = self._t(z0)
        if z.shape[0] == 1 and B > 1:
            z = z.expand(B, -1)
        ret = torch.zeros(B, device=self.dev); alive = torch.ones(B, device=self.dev)
        for t in range(acts.shape[1]):
            z = self.m.step(z, acts[:, t])
            ret = ret + alive * symexp(self.m.rew(z).squeeze(-1))
            alive = alive * torch.sigmoid(self.m.cont(z).squeeze(-1))
        return (-ret).cpu().numpy().astype(np.float64)

    def readout(self, z):
        return self.probe(z)

    def torch_fns(self):
        """Differentiable step / readout / reward for the M10 Jacobians.  The
        readout of a decoder-free model is its linear probe."""
        W = torch.as_tensor(self._probe.coef_, dtype=torch.float32, device=self.dev)
        b = torch.as_tensor(self._probe.intercept_, dtype=torch.float32, device=self.dev)
        return (lambda x, a: self.m.step(x[None], a[None])[0],
                lambda x: x @ W.T + b,
                lambda x: self.m.rew(x[None])[0, 0])
