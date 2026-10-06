"""Adapter protocol for world-model families (plan Section 2.5, rule 2).

Every family is wrapped so that the planner, the utility pipelines and the
metric library see one interface.  The names follow stable-worldmodel
(`Dynamics.encode` / `Dynamics.rollout`, `Costable.get_cost`), extended with
the members this study needs: `probe`, `sample` and `decode`.

Arrays crossing the interface are **numpy float32/float64**, batch-first.
Adapters may use torch internally; they convert at the boundary.  This keeps
the metric library testable without a GPU and makes the analytic linear-
Gaussian reference (metrics/lqg.py) a drop-in adapter.

Shapes
------
obs      (B, *obs_shape)
z        (B, D)            latent
actions  (B, H, A)         action sequence
zs       (B, H + 1, D)     rollout including the initial latent
state    (B, S)            ground-truth state, as returned by env.state()
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional
import numpy as np


@runtime_checkable
class Dynamics(Protocol):
    """Minimum every family must provide."""

    latent_dim: int
    action_dim: int

    def encode(self, obs: np.ndarray) -> np.ndarray: ...

    def step(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """One-step mean prediction.  (B, D), (B, A) -> (B, D)."""

    def rollout(self, z0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Free-running rollout.  (B, D), (B, H, A) -> (B, H + 1, D)."""


class BaseAdapter:
    """Default implementations.  Subclasses override what they support."""

    latent_dim: int = 0
    action_dim: int = 0
    family: str = "base"
    #: filled in by the registry
    checkpoint_id: str = ""

    # --- required ---------------------------------------------------------
    def encode(self, obs):  # pragma: no cover - abstract
        raise NotImplementedError

    def step(self, z, a):  # pragma: no cover - abstract
        raise NotImplementedError

    def encode_sequence(self, obs, actions):
        """Latents for a whole observed sequence: (N, T+1, *obs), (N, T, A)
        -> (N, T+1, D).

        This is the "encoded truth" every metric compares against.  The default
        encodes each frame independently, which is right for Markovian encoders
        (TD-MPC2, LeWM, DINO-WM, the linear reference).  Recurrent families
        (the DreamerV3 RSSM) override it with posterior filtering, because their
        latent at time t depends on the whole history, not on frame t alone.
        """
        obs = np.asarray(obs)
        return np.stack([np.asarray(self.encode(obs[:, t]))
                         for t in range(obs.shape[1])], axis=1)

    def rollout(self, z0, actions):
        z0 = np.asarray(z0)
        actions = np.asarray(actions)
        B, H, _ = actions.shape
        out = np.empty((B, H + 1, self.latent_dim), dtype=np.float64)
        out[:, 0] = z0
        z = z0
        for t in range(H):
            z = self.step(z, actions[:, t])
            out[:, t + 1] = z
        return out

    # --- optional ---------------------------------------------------------
    has_decoder: bool = False
    has_native_reward: bool = False
    is_stochastic: bool = False

    def decode(self, z):
        raise NotImplementedError(f"{self.family} has no decoder")

    def reward(self, z, a=None):
        """Predicted reward.  Native head, or a head fitted on frozen latents
        (plan Section 4, 'Reward heads')."""
        raise NotImplementedError(f"{self.family} has no reward head")

    def sample(self, z, a, n):
        """n stochastic successor samples.  (B, D), (B, A) -> (n, B, D).

        Deterministic families raise; metric M9 then falls back to the
        across-seed variant (plan Section 6.1, M9)."""
        raise NotImplementedError(f"{self.family} is deterministic")

    def probe(self, z):
        """Map latent to ground-truth state using the fitted probe
        (plan Appendix B).  Set by `attach_probe`."""
        if getattr(self, "_probe", None) is None:
            # optional capability, like decode/sample: absent until attached
            raise NotImplementedError("no probe attached; call attach_probe() first")
        return self._probe.predict(np.asarray(z))

    _probe = None

    def attach_probe(self, probe):
        self._probe = probe
        return self

    @property
    def has_probe(self) -> bool:
        return self._probe is not None


class ConformanceError(AssertionError):
    pass


def check_precision(device_capability=None, autocast_dtype=None):
    """Plan Amendment A3.  Volta (sm_70) reports bf16 as supported but emulates
    it at about a tenth of fp16 throughput (9.6 vs 92.1 TFLOP/s measured), so
    any adapter running bf16 there is a silent nine-fold slowdown."""
    if device_capability is None:
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
            device_capability = torch.cuda.get_device_capability(0)
        except ImportError:
            return "cpu"
    if autocast_dtype is not None and "bfloat16" in str(autocast_dtype) \
            and tuple(device_capability) < (8, 0):
        raise ConformanceError(
            f"bf16 autocast on sm_{device_capability[0]}{device_capability[1]}: "
            "emulated and ~10x slower than fp16 on this GPU; use float16")
    return "fp16" if tuple(device_capability) < (8, 0) else "bf16"


def check_conformance(adapter: BaseAdapter, obs_batch: Optional[np.ndarray] = None,
                      batch: int = 4, horizon: int = 5, strict: bool = True) -> dict:
    """Conformance suite every adapter must pass before it is used in an
    experiment (plan Section 2.5, rule 2; Section 12).

    Returns a dict of capability flags.  Raises ConformanceError on a
    violation of a required property.
    """
    rng = np.random.default_rng(0)
    D, A = adapter.latent_dim, adapter.action_dim
    report = {"family": adapter.family, "latent_dim": D, "action_dim": A}

    def req(cond, msg):
        if not cond:
            raise ConformanceError(f"[{adapter.family}] {msg}")

    req(D > 0 and A > 0, "latent_dim and action_dim must be positive")

    if obs_batch is not None:
        z = adapter.encode(obs_batch)
        z = np.asarray(z)
        req(z.ndim == 2 and z.shape[1] == D,
            f"encode must return (B, {D}), got {z.shape}")
        req(z.shape[0] == len(obs_batch), "encode must preserve batch size")
        # determinism of the encoder
        z2 = np.asarray(adapter.encode(obs_batch))
        req(np.allclose(z, z2, atol=1e-5), "encode is not deterministic")
        report["encode"] = True
    else:
        z = rng.standard_normal((batch, D))
        report["encode"] = False

    a = rng.standard_normal((z.shape[0], A))
    z1 = np.asarray(adapter.step(z, a))
    req(z1.shape == z.shape, f"step must return (B, {D}), got {z1.shape}")
    req(np.all(np.isfinite(z1)), "step returned non-finite values")

    acts = rng.standard_normal((z.shape[0], horizon, A))
    zs = np.asarray(adapter.rollout(z, acts))
    req(zs.shape == (z.shape[0], horizon + 1, D),
        f"rollout must return (B, H+1, {D}), got {zs.shape}")
    req(np.allclose(zs[:, 0], z, atol=1e-6),
        "rollout[:, 0] must be the initial latent")
    # rollout must agree with iterated step (the metric library relies on it)
    req(np.allclose(zs[:, 1], np.asarray(adapter.step(z, acts[:, 0])), atol=1e-6),
        "rollout disagrees with step at t=1")

    for name, fn in (("decode", lambda: adapter.decode(z)),
                     ("reward", lambda: adapter.reward(z)),
                     ("sample", lambda: adapter.sample(z, a, 3)),
                     ("probe", lambda: adapter.probe(z))):
        try:
            out = np.asarray(fn())
            report[name] = True
            if name == "sample":
                req(out.shape[:2] == (3, z.shape[0]),
                    f"sample must return (n, B, D), got {out.shape}")
                if strict:
                    req(out.std(axis=0).max() > 0,
                        "sample returned identical draws on a stochastic model")
            elif name == "reward":
                req(out.shape[0] == z.shape[0], "reward must be per-item")
        except NotImplementedError:
            report[name] = False
    return report
