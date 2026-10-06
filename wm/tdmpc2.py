"""TD-MPC2 family for the DMControl main grid (plan Section 4), trained offline
on D with TD-MPC2's own objective, from 3 stacked 64x64 frames.

Everything follows github.com/nicklashansen/tdmpc2 at commit e9f5932
(tdmpc2.py, common/world_model.py, common/layers.py, common/math.py,
common/scale.py, common/init.py, config.yaml; envs/dmcontrol.py for pixels):

  encoder   ShiftAug -> /255 - 0.5 -> 4 convs (7/2, 5/2, 3/2, 3/1, ReLU) ->
            flatten -> SimNorm(8); input 3 frames stacked on channels, oldest
            first, the stack filled with the first frame at reset
  dynamics  mlp(latent + action, 2 x mlp_dim, latent, act=SimNorm)
  reward    mlp(latent + action, 2 x mlp_dim, 101 bins), two-hot in symlog
            space on [-10, 10]; r_t is predicted from (z_t, a_t)
  pi        Gaussian policy prior mlp(latent, 2 x mlp_dim, 2 x action)
  Qs        5 Q-functions (2 for the 1M size), dropout 0.01, EMA targets
  update    horizon 3, rho 0.5, consistency 20, reward 0.1, value 0.1,
            Adam 3e-4 (encoder x 0.3), grad clip 20, tau 0.01, batch 256,
            discount 0.99 for 500-decision episodes, entropy 1e-4,
            RunningScale on the policy's Q

Deviations, for the pre-registration log:
 1. Offline.  Every update samples D_train (plan rule 1); TD-MPC2's own
    multitask models (mt30, mt80) were trained offline on fixed data too.
 2. ShiftAug is applied in updates only.  Upstream also applies it when
    acting, which would make encode() random; the conformance suite requires
    a deterministic encoder.  Training is unchanged: both the online and the
    target encodings are augmented, each with its own draw, as upstream.
 3. Size variants 1M and 19M take the MLP sizes of MODEL_SIZE
    (common/__init__.py).  The conv encoder's channels are latent_dim / 16, so
    its flattened 4 x 4 output is the latent, as in the 5M pixel model
    (32 channels -> 512).  Upstream only runs pixels at 5M.
 4. The Q ensemble is a ModuleList evaluated in a loop instead of torch.vmap
    over stacked parameters: the same function and per-member dropout draws.
 5. fp32 training, as upstream (no autocast).
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wm.protocol import BaseAdapter

MODEL_SIZE = {1: dict(mlp_dim=384, latent_dim=128, num_q=2),         # common/__init__.py
              5: dict(mlp_dim=512, latent_dim=512, num_q=5),
              19: dict(mlp_dim=1024, latent_dim=768, num_q=5)}
NUM_BINS, VMIN, VMAX = 101, -10.0, 10.0
BIN_SIZE = (VMAX - VMIN) / (NUM_BINS - 1)
LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0
CFG = dict(horizon=3, rho=0.5, consistency_coef=20.0, reward_coef=0.1, value_coef=0.1,
           lr=3e-4, enc_lr_scale=0.3, grad_clip_norm=20.0, tau=0.01, discount=0.99,
           entropy_coef=1e-4, batch_size=256, simnorm_dim=8, dropout=0.01, num_frames=3)


# --- common/math.py ---------------------------------------------------------
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x):
    """(..., 1) scalars -> (..., NUM_BINS) soft two-hot targets in symlog space."""
    x = torch.clamp(symlog(x), VMIN, VMAX).squeeze(-1)
    idx = torch.floor((x - VMIN) / BIN_SIZE)
    off = ((x - VMIN) / BIN_SIZE - idx).unsqueeze(-1)
    idx = idx.long().unsqueeze(-1)
    out = torch.zeros(*x.shape, NUM_BINS, device=x.device, dtype=x.dtype)
    out = out.scatter(-1, idx, 1 - off)
    return out.scatter(-1, (idx + 1) % NUM_BINS, off)


def two_hot_inv(logits):
    bins = torch.linspace(VMIN, VMAX, NUM_BINS, device=logits.device, dtype=logits.dtype)
    return symexp((F.softmax(logits, dim=-1) * bins).sum(-1, keepdim=True))


def soft_ce(pred, target):
    return -(two_hot(target) * F.log_softmax(pred, dim=-1)).sum(-1, keepdim=True)


def gaussian_logprob(eps, log_std):
    return (-0.5 * eps.pow(2) - log_std - 0.9189385175704956).sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
    mu, pi = torch.tanh(mu), torch.tanh(pi)
    return mu, pi, log_pi - torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)


# --- common/layers.py ---------------------------------------------------------
class ShiftAug(nn.Module):
    """Random shift by up to `pad` pixels (DrQ-v2), as upstream."""

    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        x = F.pad(x, (self.pad,) * 4, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        grid = torch.cat([ar, ar.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        grid = grid + shift * (2.0 / (h + 2 * self.pad))
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class SimNorm(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shp = x.shape
        return F.softmax(x.view(*shp[:-1], -1, self.dim), dim=-1).view(shp)


class NormedLinear(nn.Linear):
    def __init__(self, i, o, dropout=0.0, act=None):
        super().__init__(i, o)
        self.ln = nn.LayerNorm(o)
        self.act = act if act is not None else nn.Mish()
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))


def mlp(i, dims, o, act=None, dropout=0.0):
    d = [i] + list(dims) + [o]
    layers = [NormedLinear(d[k], d[k + 1], dropout=dropout * (k == 0)) for k in range(len(d) - 2)]
    layers.append(NormedLinear(d[-2], d[-1], act=act) if act is not None else nn.Linear(d[-2], d[-1]))
    return nn.Sequential(*layers)


def weight_init(m):
    """common/init.py"""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class RunningScale(nn.Module):
    """common/scale.py: trimmed (5-95 %) running scale of the policy's Q."""

    def __init__(self, tau=CFG["tau"]):
        super().__init__()
        self.tau = tau
        self.register_buffer("value", torch.ones(1))
        self.register_buffer("pct", torch.tensor([5.0, 95.0]))

    def update(self, x):
        x = x.detach().flatten(1).float()
        s = torch.sort(x, dim=0).values
        pos = self.pct * (x.shape[0] - 1) / 100
        lo = torch.floor(pos); hi = torch.clamp(lo + 1, max=x.shape[0] - 1)
        w = (pos - lo).unsqueeze(1)
        p = s[lo.long()] * (1 - w) + s[hi.long()] * w
        self.value.data.lerp_(torch.clamp(p[1] - p[0], min=1.0).reshape(1), self.tau)

    def forward(self, x):
        return x / self.value


# --- common/world_model.py -----------------------------------------------------
class PixelTDMPC2(nn.Module):
    def __init__(self, act_dim, size=5):
        super().__init__()
        c = MODEL_SIZE[size]
        L, M, self.num_q = c["latent_dim"], c["mlp_dim"], c["num_q"]
        self.size, self.act_dim, self.latent_dim = size, act_dim, L
        ch, nf = L // 16, CFG["num_frames"]
        self.aug = ShiftAug(3)
        self.conv = nn.Sequential(
            nn.Conv2d(3 * nf, ch, 7, stride=2), nn.ReLU(),
            nn.Conv2d(ch, ch, 5, stride=2), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, stride=2), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, stride=1), nn.Flatten(), SimNorm(CFG["simnorm_dim"]))
        sn = SimNorm(CFG["simnorm_dim"])
        self.dynamics = mlp(L + act_dim, [M, M], L, act=sn)
        self.reward = mlp(L + act_dim, [M, M], NUM_BINS)
        self.pi = mlp(L, [M, M], 2 * act_dim)
        self.Qs = nn.ModuleList([mlp(L + act_dim, [M, M], NUM_BINS, dropout=CFG["dropout"])
                                 for _ in range(self.num_q)])
        self.apply(weight_init)
        nn.init.zeros_(self.reward[-1].weight)
        for q in self.Qs:
            nn.init.zeros_(q[-1].weight)
        self.target_Qs = copy.deepcopy(self.Qs).requires_grad_(False)
        self.register_buffer("log_std_min", torch.tensor(LOG_STD_MIN))
        self.register_buffer("log_std_dif", torch.tensor(LOG_STD_MAX - LOG_STD_MIN))

    def train(self, mode=True):
        super().train(mode)
        self.target_Qs.train(False)          # target Qs stay in eval mode, as upstream
        return self

    # -- pieces ------------------------------------------------------------
    def encode(self, x, aug=False):
        """(..., 3*nf, 64, 64) uint8/float stacks in 0-255 -> (..., latent)."""
        lead = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:]).float()
        if aug:
            x = self.aug(x)
        return self.conv(x / 255.0 - 0.5).reshape(*lead, -1)

    def next(self, z, a):
        return self.dynamics(torch.cat([z, a], -1))

    def reward_logits(self, z, a):
        return self.reward(torch.cat([z, a], -1))

    def pi_sample(self, z):
        mean, log_std = self.pi(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * self.log_std_dif * (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mean)
        log_prob = gaussian_logprob(eps, log_std)
        scaled_log_prob = log_prob * eps.shape[-1]
        action = mean + eps * log_std.exp()
        mean, action, log_prob = squash(mean, action, log_prob)
        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        return action, mean, -log_prob * entropy_scale

    def Q(self, z, a, return_type="min", target=False, detach=False):
        za = torch.cat([z, a], -1)
        nets = self.target_Qs if target else self.Qs
        if detach:
            outs = [torch.func.functional_call(q, {k: v.detach() for k, v in q.named_parameters()}, (za,))
                    for q in nets]
        else:
            outs = [q(za) for q in nets]
        out = torch.stack(outs)
        if return_type == "all":
            return out
        idx = torch.randperm(self.num_q, device=out.device)[:2]
        Qv = two_hot_inv(out[idx])
        return Qv.min(0).values if return_type == "min" else Qv.sum(0) / 2

    @torch.no_grad()
    def soft_update_target_Q(self, tau=CFG["tau"]):
        for p, tp in zip(self.Qs.parameters(), self.target_Qs.parameters()):
            tp.data.lerp_(p.data, tau)


def make_optimizers(model):
    """tdmpc2.py: one Adam for the model (encoder at 0.3 x lr), one for pi."""
    lr = CFG["lr"]
    opt = torch.optim.Adam([
        {"params": model.conv.parameters(), "lr": lr * CFG["enc_lr_scale"]},
        {"params": model.dynamics.parameters()},
        {"params": model.reward.parameters()},
        {"params": model.Qs.parameters()}], lr=lr)
    pi_opt = torch.optim.Adam(model.pi.parameters(), lr=lr, eps=1e-5)
    return opt, pi_opt


def model_losses(model, obs, action, reward, aug):
    """tdmpc2.py _update, model part.  obs (H+1, B, 3nf, 64, 64), action (H, B, A),
    reward (H, B, 1).  Returns (total, zs, info)."""
    H, rho = action.shape[0], CFG["rho"]
    with torch.no_grad():
        next_z = model.encode(obs[1:], aug=aug)
        pi_a, _, _ = model.pi_sample(next_z)
        td = reward + CFG["discount"] * model.Q(next_z, pi_a, "min", target=True)
    z = model.encode(obs[0], aug=aug)
    zs, cons = [z], 0.0
    for t in range(H):
        z = model.next(z, action[t])
        cons = cons + F.mse_loss(z, next_z[t]) * rho ** t
        zs.append(z)
    zs = torch.stack(zs)
    qs = model.Q(zs[:-1], action, "all")                     # (num_q, H, B, bins)
    rp = model.reward_logits(zs[:-1], action)                # (H, B, bins)
    rl, vl = 0.0, 0.0
    for t in range(H):
        rl = rl + soft_ce(rp[t], reward[t]).mean() * rho ** t
        for q in range(model.num_q):
            vl = vl + soft_ce(qs[q, t], td[t]).mean() * rho ** t
    cons, rl, vl = cons / H, rl / H, vl / (H * model.num_q)
    total = CFG["consistency_coef"] * cons + CFG["reward_coef"] * rl + CFG["value_coef"] * vl
    return total, zs, {"consistency": float(cons), "reward": float(rl), "value": float(vl),
                       "wm": float(CFG["consistency_coef"] * cons + CFG["reward_coef"] * rl),
                       "loss": float(total)}


def update(model, opt, pi_opt, scale, obs, action, reward):
    """One TD-MPC2 iteration: model update, policy update, target-Q update."""
    model.train()
    total, zs, info = model_losses(model, obs, action, reward, aug=True)
    total.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])
    opt.step(); opt.zero_grad(set_to_none=True)
    # update_pi
    zs = zs.detach()
    a, _, scaled_entropy = model.pi_sample(zs)
    qs = model.Q(zs, a, "avg", detach=True)
    scale.update(qs[0])
    qs = scale(qs)
    rho = torch.pow(CFG["rho"], torch.arange(len(qs), device=qs.device))
    pi_loss = (-(CFG["entropy_coef"] * scaled_entropy + qs).mean(dim=(1, 2)) * rho).mean()
    pi_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.pi.parameters(), CFG["grad_clip_norm"])
    pi_opt.step(); pi_opt.zero_grad(set_to_none=True)
    model.soft_update_target_Q()
    model.eval()
    info.update(pi_loss=float(pi_loss), grad_norm=float(gn), pi_scale=float(scale.value))
    return info


@torch.no_grad()
def val_losses(model, obs, action, reward):
    """The training objective on held-out windows, without augmentation or
    dropout (the checkpoint's validation loss)."""
    model.eval()
    _, _, info = model_losses(model, obs, action, reward, aug=False)
    return info


# --- adapter -------------------------------------------------------------------
def stack_frames(obs, nf=CFG["num_frames"]):
    """(N, T+1, 64, 64, 3) frames -> (N, T+1, 3nf, 64, 64) stacks, oldest frame
    first, the first frame repeated until the history fills (Pixels wrapper)."""
    obs = np.asarray(obs)
    T1 = obs.shape[1]
    idx = np.arange(T1)
    parts = [obs[:, np.clip(idx - (nf - 1 - k), 0, None)] for k in range(nf)]
    st = np.stack(parts, 2)                                  # (N, T+1, nf, 64, 64, 3)
    return st.transpose(0, 1, 2, 5, 3, 4).reshape(obs.shape[0], T1, 3 * nf, 64, 64)


class TDMPC2Adapter(BaseAdapter):
    """The latent is TD-MPC2's z of the 3-frame stack.  encode() treats a lone
    frame as a fresh episode (the stack filled with it, as at reset);
    encode_sequence() stacks each frame with its two predecessors; filter_step()
    keeps the last frames of each row of the batch it was last given by
    encode(), which is how U1 advances its lockstep episodes.  Rewards are
    predicted from (z_t, a_t), so `reward_needs_action` tells the U2 loop to
    score a step before taking it."""
    family = "tdmpc2"
    has_decoder = False
    has_native_reward = True
    is_stochastic = False
    reward_needs_action = True

    def __init__(self, model: PixelTDMPC2, device="cpu"):
        self.m = model.to(device).eval(); self.dev = device
        self.latent_dim, self.action_dim = model.latent_dim, model.act_dim
        self.nf = CFG["num_frames"]
        self._hist = None
        self._probe = None

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

    @torch.no_grad()
    def _enc(self, stacks, chunk=1024):
        x = torch.as_tensor(np.ascontiguousarray(stacks), device=self.dev)
        flat = x.reshape(-1, *x.shape[-3:])
        z = torch.cat([self.m.encode(flat[i:i + chunk]) for i in range(0, len(flat), chunk)])
        return z.reshape(*x.shape[:-3], -1).cpu().numpy().astype(np.float64)

    def encode(self, obs):
        obs = np.asarray(obs)
        self._hist = np.repeat(obs[:, None], self.nf, 1)          # (B, nf, 64, 64, 3)
        return self._enc(stack_frames(self._hist)[:, -1])

    def encode_sequence(self, obs, actions):
        return self._enc(stack_frames(obs))

    def filter_step(self, z, a, obs):
        obs = np.asarray(obs)
        if self._hist is None or len(self._hist) != len(obs):
            raise RuntimeError("filter_step needs the frames of the batch last passed to encode()")
        self._hist = np.concatenate([self._hist[:, 1:], obs[:, None]], 1)
        return self._enc(stack_frames(self._hist)[:, -1])

    @torch.no_grad()
    def step(self, z, a):
        return self.m.next(self._t(z), self._t(a)).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def rollout(self, z0, actions):
        acts = self._t(actions); z = self._t(z0); out = [z]
        for t in range(acts.shape[1]):
            z = self.m.next(z, acts[:, t]); out.append(z)
        return torch.stack(out, 1).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def reward(self, z, a=None):
        """r(z_t, a_t); without an action the zero action is used."""
        z = self._t(z)
        a = torch.zeros(z.shape[0], self.action_dim, device=self.dev) if a is None else self._t(a)
        return two_hot_inv(self.m.reward_logits(z, a)).squeeze(-1).cpu().numpy().astype(np.float64)

    def cont_prob(self, z):                 # DMControl episodes never terminate
        return np.ones(len(np.asarray(z)))

    @torch.no_grad()
    def get_cost(self, z0, actions):
        """Negated sum over the horizon of predicted rewards r(z_t, a_t); no
        value bootstrap (plan Section 5.1)."""
        acts = self._t(actions); B = acts.shape[0]; z = self._t(z0)
        if z.shape[0] == 1 and B > 1:
            z = z.expand(B, -1)
        ret = torch.zeros(B, device=self.dev)
        for t in range(acts.shape[1]):
            ret = ret + two_hot_inv(self.m.reward_logits(z, acts[:, t]).float()).squeeze(-1)
            z = self.m.next(z, acts[:, t])
        return (-ret).cpu().numpy().astype(np.float64)

    def readout(self, z):
        return self.probe(z)

    def torch_fns(self):
        """Differentiable step / readout / reward for the M10 Jacobians; the
        reward is r(z, 0), the action at which the Jacobians are taken."""
        W = torch.as_tensor(self._probe.coef_, dtype=torch.float32, device=self.dev)
        b = torch.as_tensor(self._probe.intercept_, dtype=torch.float32, device=self.dev)
        a0 = torch.zeros(1, self.action_dim, device=self.dev)
        return (lambda x, a: self.m.next(x[None], a[None])[0],
                lambda x: x @ W.T + b,
                lambda x: two_hot_inv(self.m.reward_logits(x[None], a0))[0, 0])
