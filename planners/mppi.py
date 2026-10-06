"""Family-agnostic MPPI planner (plan Section 5, U1; Appendix A).

One planner for every world-model family: it consumes only `rollout` and
`reward` from the adapter, so a family's native planner is never used for U1.
Trajectory cost is the sum of predicted rewards over H_plan with no value
bootstrap, which keeps the objective identical across families.

Settings are frozen per environment in Appendix A; `PlanConfig` mirrors
stable-worldmodel's so external checkpoints drop in unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PlanConfig:
    horizon: int = 16            # H_plan, in environment steps
    receding_horizon: int = 1    # k, replanning interval
    population: int = 512
    iterations: int = 6
    temperature: float = 0.5
    elite_frac: float = 0.1
    init_std: float = 1.0
    min_std: float = 0.05
    momentum: float = 0.1
    action_block: int = 1        # LeWM plans in blocks of 5 env steps
    warm_start: bool = True
    action_low: float = -np.inf
    action_high: float = np.inf
    method: str = "mppi"         # "mppi" (softmax weights) or "cem" (elites)


class MPPI:
    def __init__(self, adapter, cfg: PlanConfig, action_dim: int, seed: int = 0):
        self.ad, self.cfg, self.A = adapter, cfg, action_dim
        self.rng = np.random.default_rng(seed)
        self._mu = None

    def reset(self):
        self._mu = None

    def _cost(self, z0, actions):
        """Negated predicted return.  Uses the adapter's fused
        `get_cost` (stable-worldmodel's Costable protocol) when it has one."""
        if hasattr(self.ad, "get_cost"):
            return np.asarray(self.ad.get_cost(z0, actions))
        B = actions.shape[0]
        z = np.repeat(np.atleast_2d(z0), B, axis=0) if np.atleast_2d(z0).shape[0] == 1 else z0
        traj = np.asarray(self.ad.rollout(z, actions))
        r = np.zeros(B)
        for t in range(1, traj.shape[1]):
            r += np.atleast_1d(np.asarray(self.ad.reward(traj[:, t])))
        return -r

    def plan(self, z0):
        """Return the action sequence to execute: (receding_horizon, A)."""
        c = self.cfg
        H = int(np.ceil(c.horizon / c.action_block))
        mu = np.zeros((H, self.A)) if (self._mu is None or not c.warm_start) else self._mu
        std = np.full((H, self.A), c.init_std)
        for _ in range(c.iterations):
            noise = self.rng.standard_normal((c.population, H, self.A))
            seq = np.clip(mu[None] + std[None] * noise, c.action_low, c.action_high)
            env_seq = np.repeat(seq, c.action_block, axis=1)[:, :c.horizon]
            J = self._cost(z0, env_seq)
            J = np.nan_to_num(J, nan=1e9, posinf=1e9)
            if c.method == "cem":
                n_elite = max(1, int(round(c.elite_frac * c.population)))
                w = np.zeros_like(J); w[np.argsort(J)[:n_elite]] = 1.0 / n_elite
            else:
                w = np.exp(-(J - J.min()) / max(c.temperature, 1e-8))
                w = w / max(w.sum(), 1e-12)
            new_mu = np.einsum("b,bha->ha", w, seq)
            new_std = np.sqrt(np.einsum("b,bha->ha", w, (seq - new_mu[None]) ** 2))
            mu = (1 - c.momentum) * new_mu + c.momentum * mu
            std = np.maximum(new_std, c.min_std)
        if c.warm_start:
            self._mu = np.roll(mu, -1, axis=0); self._mu[-1] = 0.0
        out = np.repeat(mu, c.action_block, axis=0)[:c.horizon]
        return out[:c.receding_horizon]
