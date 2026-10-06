"""TD-MPC2 "data agent" (plan Section 3.2): acts with a released TD-MPC2 checkpoint.

Inference-only re-implementation of what TD-MPC2 needs to act, loaded from the
official single-task, state-based, 5M-parameter checkpoints at
https://huggingface.co/nicklashansen/tdmpc2 (dmcontrol/<task>-<seed>.pt, seeds
1-3).  They are stored under artifacts/data_agents/tdmpc2_5M/; see checkpoint_path.

Two ways to act:
  * mpc=False: the policy prior's deterministic action tanh(mu).  This is what
    TD-MPC2 returns in eval mode without planning (tdmpc2/tdmpc2.py:116-120).
    It costs about 1 ms per step on a CPU.  It matches the published returns
    on cartpole-swingup and cup-catch.  On walker-walk it is far too weak (about
    470 against 981): it walks too fast and falls, under MuJoCo 2.3.1 as well
    as 3.12.
  * mpc=True: TD-MPC2's MPPI planner over the learned world model
    (tdmpc2.py:122-206), with the default settings in config.yaml:34-42.  About
    0.3 s per decision on 4 CPU threads; batching environments amortises it.
    walker-walk needs this.  DEFAULT_MPC holds the per-task choice.
Data-collection noise is N(0, noise_std^2), added to that action and then
clipped to [-1, 1].

Source citations are to github.com/nicklashansen/tdmpc2 at commit e9f5932
(HEAD on 2026-09-26).  "b67b21c" is the first public commit (2023-10-25), the
code closest to the checkpoints, whose metadata is dated 2023-06-03.

Observation contract (checked against both commits):
  * Order: TD-MPC2 flattens the dm_control observation OrderedDict in its own
    order, `np.concatenate([v.flatten() for v in obs.values()])`
    (envs/dmcontrol.py:47-49; b67b21c envs/dmcontrol.py:165-166).  That is the
    same order as `envs/dmc.py` builds `proprio`, so `proprio` goes in as is.
    Evidence: the reward head predicts the true per-step reward from our
    observations (correlation 1.00 / 0.99 / 0.97 on the three tasks).
  * Normalisation: none.  Raw float32 state goes straight into the encoder
    (common/world_model.py:103-112).  The only input normalisation in TD-MPC2
    is for pixels (common/layers.py:62-71).  `RunningScale` rescales Q-values
    in the policy loss, not observations (tdmpc2.py:221-222).  The exception
    is pre-release checkpoints, which apply symlog first; see load_state_dict.
  * Actions: the wrapper rescales to [-1, 1] (envs/dmcontrol.py:106).  That is
    the identity for cartpole, walker and ball_in_cup, whose native bounds
    are [-1, 1].  Action repeat is 2 (envs/dmcontrol.py:57; b67b21c line 196),
    and episodes are 500 decisions (envs/dmcontrol.py:110).  Both match the
    DMCEnv defaults.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "data_agents" / "tdmpc2_5M"
HF_URL = "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/dmcontrol/{task}-{seed}.pt"
# Seed 2 for cartpole because cartpole-swingup-1.pt uses the pre-release layout
# (see load_state_dict).  It does work with the symlog input: 867 against a
# published 866 for that seed.
DEFAULT_SEED = {"cartpole-swingup": 2, "walker-walk": 1, "cup-catch": 1}
DEFAULT_MPC = {"cartpole-swingup": False, "walker-walk": True, "cup-catch": False}

# MPPI settings: config.yaml:34-42.  The discount follows tdmpc2.py:57-70 with
# 500-step episodes: frac = 500/5, (frac-1)/frac = 0.99, clipped to [0.95, 0.995].
PLAN_CFG = dict(horizon=3, iterations=6, num_samples=512, num_elites=64, num_pi_trajs=24,
                min_std=0.05, max_std=2.0, temperature=0.5, discount=0.99)
LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0        # config.yaml:45-46
NUM_BINS, VMIN, VMAX = 101, -10.0, 10.0      # config.yaml:50-52 (two-hot reward and Q heads)
_QNAMES = ["weight", "bias", "ln.weight", "ln.bias"]


def checkpoint_path(task: str, seed: int | None = None) -> Path:
    """Local path of the released checkpoint <task>-<seed>.pt (downloaded from HF_URL)."""
    seed = DEFAULT_SEED[task] if seed is None else seed
    return CKPT_DIR / f"{task}-{seed}.pt"


# --- building blocks, as in tdmpc2/common/layers.py and common/math.py -----
class SimNorm(nn.Module):
    """Softmax over consecutive groups of `dim` latent units.
    common/layers.py:74-88; group size simnorm_dim=8 (config.yaml:64)."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shp = x.shape
        return F.softmax(x.view(*shp[:-1], -1, self.dim), dim=-1).view(shp)


class NormedLinear(nn.Linear):
    """Linear -> LayerNorm -> activation (Mish unless given).  common/layers.py:94-111.
    Dropout is omitted: only the Q-functions use it, and the model acts in eval
    mode (tdmpc2.py:32, 319)."""

    def __init__(self, in_features, out_features, act: nn.Module | None = None):
        super().__init__(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)       # default eps=1e-5, as upstream
        self.act = act if act is not None else nn.Mish()

    def forward(self, x):
        return self.act(self.ln(super().forward(x)))


def _mlp(dims: list[int], act: nn.Module | None = None) -> nn.Sequential:
    """common/layers.py:121-133: NormedLinear+Mish for the hidden layers, then a
    NormedLinear with `act` when `act` is given, or a plain Linear when it is not."""
    layers = [NormedLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 2)]
    layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act is not None
                  else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def symlog(x):
    """common/math.py:42-47."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def two_hot_inv(logits):
    """Soft two-hot logits to a scalar: softmax over bins, expectation, symexp.
    common/math.py:74-83."""
    bins = torch.linspace(VMIN, VMAX, NUM_BINS, device=logits.device, dtype=logits.dtype)
    x = (F.softmax(logits, dim=-1) * bins).sum(-1, keepdim=True)
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# --- checkpoint handling -----------------------------------------------------
def _sub(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _convert_flat_layout(sd: dict) -> dict:
    """Convert the pre-release layout (a flat nn.Sequential of Linear, LayerNorm and
    Mish) to the NormedLinear layout: each Linear takes the next LayerNorm as `.ln`.
    This is the renaming the TD-MPC2 author gives in tdmpc2 GitHub issue #23."""
    out = dict(sd)
    for prefix in ("_encoder.state.", "_dynamics.", "_reward.", "_pi."):
        keys = [k for k in sd if k.startswith(prefix)]
        for k in keys:
            del out[k]
        mods = sorted({k[len(prefix):].rsplit(".", 1)[0] for k in keys},
                      key=lambda m: [int(p) for p in m.split(".")])
        layer = -1
        for m in mods:
            w, b = sd[f"{prefix}{m}.weight"], sd[f"{prefix}{m}.bias"]
            if w.ndim == 2:                                # Linear starts a new layer
                layer += 1
                out[f"{prefix}{layer}.weight"], out[f"{prefix}{layer}.bias"] = w, b
            else:                                          # LayerNorm of the last Linear
                out[f"{prefix}{layer}.ln.weight"], out[f"{prefix}{layer}.ln.bias"] = w, b
    return out


def load_state_dict(path) -> tuple[dict, dict, bool]:
    """Return (state_dict in the NormedLinear layout, metadata, input_symlog).

    A few released checkpoints predate a refactor and use a flat layout
    (tdmpc2 issues #23 and #78; here only cartpole-swingup-1).  In that layout
    `_encoder.state.0` is a module without parameters.  The author's renaming
    drops it, and the resulting agent fails (cartpole return 157).  The module
    is symlog of the observation, identified empirically:
      * Of six candidate input maps, symlog gives the lowest reward-head error
        on cartpole transitions.
      * With symlog the agent reaches 867, against a published final of 866.2
        for seed 1 (results/tdmpc2/cartpole-swingup.csv).
    Only the policy prior has been verified on this layout.  The world model
    fits worse than on seed 2 (latent consistency 0.14 against 0.0005), so do
    not plan with this checkpoint.
    """
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck      # tdmpc2.py:79 saves {"model": ...}
    meta = ck.get("metadata", {})
    if "_task_emb.weight" in sd:
        raise NotImplementedError(
            f"{path} is a multi-task checkpoint (task embedding + zero-padded obs); "
            "only single-task checkpoints are supported.")
    flat = "_encoder.state.0.weight" not in sd
    if flat:
        sd = _convert_flat_layout(sd)
    return sd, meta, flat


def _q_state_dicts(sd: dict) -> list[dict]:
    """Split the vectorised Q ensemble into one state_dict per Q-function.  Two layouts:
    * released checkpoints: `_Qs.params.<n>`, stacked over the ensemble, where n
      indexes layer n//4 and tensor [weight, bias, ln.weight, ln.bias][n%4]
      (common/layers.py:175-193, api_model_conversion);
    * current code: `_Qs.params.<layer>.<name>` (tensordict, common/world_model.py:30).
    """
    q = {k: v for k, v in _sub(sd, "_Qs.params.").items() if "__" not in k}
    per_name = {}
    for k, v in q.items():
        name = f"{int(k) // 4}.{_QNAMES[int(k) % 4]}" if k.isdigit() else k
        per_name[name] = v
    num_q = next(iter(per_name.values())).shape[0]
    return [{n: v[i] for n, v in per_name.items()} for i in range(num_q)]


# --- the agent ----------------------------------------------------------------
class TDMPC2Agent:
    """Acts with a released single-task, state-based TD-MPC2 checkpoint.

    act(obs, noise_std, rng) takes one (obs_dim,) or a batch (N, obs_dim) of
    DMCEnv `proprio` and returns float32 actions in [-1, 1] of shape
    (act_dim,) or (N, act_dim).  With mpc=True, call reset() at the start of
    every episode, because the planner warm-starts from its previous plan
    (tdmpc2.py:166-167, 205).
    """

    def __init__(self, path, obs_dim: int, act_dim: int, mpc: bool = False,
                 device: str = "cpu", plan_cfg: dict | None = None):
        sd, self.metadata, self.input_symlog = load_state_dict(path)
        self.path, self.mpc = Path(path), mpc
        self.device = torch.device(device)
        # Sizes come from the weights, not a config.  For reference,
        # common/__init__.py:1-24 lists 5M as enc_dim 256, mlp_dim 512,
        # latent_dim 512 and num_enc_layers 2.
        enc_dims = self._dims(sd, "_encoder.state.")
        pi_dims = self._dims(sd, "_pi.")
        if enc_dims[0] != obs_dim:
            raise ValueError(f"{path}: encoder expects obs_dim={enc_dims[0]}, got {obs_dim}")
        if pi_dims[-1] != 2 * act_dim:
            raise ValueError(f"{path}: policy outputs act_dim={pi_dims[-1] // 2}, got {act_dim}")
        self.obs_dim, self.act_dim, self.latent_dim = obs_dim, act_dim, enc_dims[-1]

        # Encoder, state branch: common/layers.py:158-159, i.e.
        # mlp(obs_dim, max(num_enc_layers-1, 1)*[enc_dim], latent_dim, act=SimNorm).
        self.encoder = _mlp(enc_dims, act=SimNorm(8))
        self.encoder.load_state_dict(_sub(sd, "_encoder.state."))
        # Policy prior: mlp(latent_dim, 2*[mlp_dim], 2*action_dim), common/world_model.py:29.
        self.pi = _mlp(pi_dims)
        self.pi.load_state_dict(_sub(sd, "_pi."))
        nets = [self.encoder, self.pi]
        if mpc:
            if self.input_symlog:
                raise ValueError(f"{path}: planning is unverified for pre-release checkpoints")
            # common/world_model.py:26-30: dynamics ends in SimNorm; the reward
            # head and every Q head end in a Linear onto NUM_BINS bins.
            self.dynamics = _mlp(self._dims(sd, "_dynamics."), act=SimNorm(8))
            self.dynamics.load_state_dict(_sub(sd, "_dynamics."))
            self.reward = _mlp(self._dims(sd, "_reward."))
            self.reward.load_state_dict(_sub(sd, "_reward."))
            qsd = _q_state_dicts(sd)
            q_dims = [qsd[0]["0.weight"].shape[1]] + \
                     [qsd[0][f"{i}.weight"].shape[0] for i in range(3)]
            self.Qs = nn.ModuleList([_mlp(q_dims) for _ in qsd])
            for q, s in zip(self.Qs, qsd):
                q.load_state_dict(s)
            nets += [self.dynamics, self.reward, self.Qs]
        for m in nets:
            m.to(self.device).eval().requires_grad_(False)
        self.cfg = dict(PLAN_CFG, **(plan_cfg or {}))
        self._gen = torch.Generator(device=self.device)
        self._gen.seed()
        self.reset()

    @staticmethod
    def _dims(sd, prefix):
        n = 1 + max(int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix))
        return [sd[f"{prefix}0.weight"].shape[1]] + [sd[f"{prefix}{i}.weight"].shape[0] for i in range(n)]

    def reset(self):
        """Start a new episode: the planner's warm start is cleared."""
        self._prev_mean = None

    # --- model pieces (common/world_model.py) --------------------------------
    def _encode(self, obs: np.ndarray):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.encoder(symlog(x) if self.input_symlog else x)     # world_model.py:103-112

    def _pi(self, z, gen=None):
        """Returns (tanh(mu), tanh(mu + std * eps)), from world_model.py:144-184.
        The second output is a sample from the prior; it is None without `gen`."""
        mu, log_std = self.pi(z).chunk(2, dim=-1)                       # world_model.py:154
        if gen is None:
            return torch.tanh(mu), None
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1)  # math.py:12-13
        eps = torch.randn(mu.shape, generator=gen, device=mu.device)
        return torch.tanh(mu), torch.tanh(mu + eps * log_std.exp())    # math.py:23-26

    def _q_avg(self, z, a, gen):
        """Average of 2 of the Q-functions drawn at random: world_model.py:212-216,
        return_type='avg'.  Only the 2 drawn heads are evaluated; TD-MPC2 runs all
        of them first, which gives the same value.  One draw is shared by the batch."""
        za = torch.cat([z, a], dim=-1)
        idx = torch.randperm(len(self.Qs), generator=gen, device=self.device)[:2].tolist()
        return sum(two_hot_inv(self.Qs[i](za)) for i in idx) / 2

    @torch.no_grad()
    def _estimate_value(self, z, actions, gen):
        """tdmpc2.py:122-136 (not episodic, so there is no termination head)."""
        G, discount = 0, 1.0
        for t in range(self.cfg["horizon"]):
            za = torch.cat([z, actions[t]], dim=-1)
            G = G + discount * two_hot_inv(self.reward(za))
            z = self.dynamics(za)
            discount = discount * self.cfg["discount"]
        _, a = self._pi(z, gen)
        return G + discount * self._q_avg(z, a, gen)

    @torch.no_grad()
    def _plan(self, obs: np.ndarray, gen) -> torch.Tensor:
        """MPPI, tdmpc2.py:138-206, with eval_mode=True, batched over N observations.
        Each observation has its own sample population and warm start."""
        c = self.cfg
        H, S, E, P, A = c["horizon"], c["num_samples"], c["num_elites"], c["num_pi_trajs"], self.act_dim
        N = obs.shape[0]
        z0 = self._encode(obs)                                            # (N, L)
        # Trajectories from the policy prior (tdmpc2.py:153-160).
        pi_actions = torch.empty(H, N * P, A, device=self.device)
        _z = z0.repeat_interleave(P, dim=0)
        for t in range(H - 1):
            _, pi_actions[t] = self._pi(_z, gen)
            _z = self.dynamics(torch.cat([_z, pi_actions[t]], dim=-1))
        _, pi_actions[-1] = self._pi(_z, gen)
        pi_actions = pi_actions.view(H, N, P, A)
        # Initial state and parameters (tdmpc2.py:162-170).
        z = z0.repeat_interleave(S, dim=0)                                # (N*S, L)
        mean = torch.zeros(H, N, A, device=self.device)
        std = torch.full((H, N, A), c["max_std"], device=self.device)
        if self._prev_mean is not None and self._prev_mean.shape[1] == N:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(H, N, S, A, device=self.device)
        actions[:, :, :P] = pi_actions
        # MPPI iterations (tdmpc2.py:172-197).
        for _ in range(c["iterations"]):
            r = torch.randn(H, N, S - P, A, generator=gen, device=self.device)
            actions[:, :, P:] = (mean.unsqueeze(2) + std.unsqueeze(2) * r).clamp(-1, 1)
            value = self._estimate_value(z, actions.view(H, N * S, A), gen).nan_to_num(0).view(N, S)
            elite_idxs = torch.topk(value, E, dim=1).indices                 # (N, E)
            elite_value = torch.gather(value, 1, elite_idxs)                 # (N, E)
            elite_actions = torch.gather(actions, 2, elite_idxs[None, :, :, None].expand(H, N, E, A))
            max_value = elite_value.max(1, keepdim=True).values
            score = torch.exp(c["temperature"] * (elite_value - max_value))
            score = score / score.sum(1, keepdim=True)                       # (N, E)
            w, denom = score[None, :, :, None], score.sum(1)[None, :, None] + 1e-9
            mean = (w * elite_actions).sum(2) / denom                        # (H, N, A)
            std = ((w * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2) / denom).sqrt()
            std = std.clamp(c["min_std"], c["max_std"])
        # Select an elite in proportion to its score: the Gumbel-max draw of
        # math.py:86-94 at temperature 1 is a categorical draw (tdmpc2.py:200-202).
        rand_idx = torch.multinomial(score, 1, generator=gen).squeeze(1)    # (N,)
        a = elite_actions[0, torch.arange(N, device=self.device), rand_idx]  # (N, A)
        self._prev_mean = mean                                               # tdmpc2.py:205
        return a.clamp(-1, 1)

    # --- public API ------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, noise_std: float = 0.0, rng: np.random.Generator | None = None,
            sample_prior: bool = False) -> np.ndarray:
        """Action(s) in [-1, 1] for one observation (obs_dim,) or a batch (N, obs_dim).

        The base action is the MPPI plan when mpc=True, else the prior mean
        tanh(mu).  With sample_prior=True (prior only) it is instead a draw
        tanh(mu + std * eps) from the prior's own Gaussian.  N(0, noise_std^2)
        is then added and the result clipped.  All randomness, the planner's
        included, comes from `rng`, so a given rng gives the same actions.
        """
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        obs = obs[None] if single else obs
        gen = None
        if self.mpc or sample_prior:
            gen = self._gen if rng is None else \
                torch.Generator(device=self.device).manual_seed(int(rng.integers(2 ** 63 - 1)))
        if self.mpc:
            a = self._plan(obs, gen)
        else:
            mean, sample = self._pi(self._encode(obs), gen if sample_prior else None)
            a = sample if sample_prior else mean
        a = a.cpu().numpy()
        if noise_std > 0:
            rng = np.random.default_rng() if rng is None else rng
            a = a + noise_std * rng.standard_normal(a.shape)
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        return a[0] if single else a

    def __repr__(self):
        return (f"TDMPC2Agent({self.path.name}, obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
                f"latent={self.latent_dim}, mpc={self.mpc}, input_symlog={self.input_symlog})")


def make_agent(task: str, seed: int | None = None, mpc: bool | None = None,
               device: str = "cpu") -> TDMPC2Agent:
    """Data agent for "cartpole-swingup" | "walker-walk" | "cup-catch", using the
    default seed and planning choice for the task.  Dimensions come from DMCEnv."""
    from envs.dmc import DMCEnv
    env = DMCEnv(task, render=False)
    obs_dim = env.reset(seed=0)["proprio"].shape[0]
    mpc = DEFAULT_MPC[task] if mpc is None else mpc
    return TDMPC2Agent(checkpoint_path(task, seed), obs_dim, env.act_dim, mpc=mpc, device=device)
