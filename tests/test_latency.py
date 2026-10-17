"""Gate G0 for the side paper (sidepaper/EXPERIMENT_PLAN_SIDE.md, Section 4).

1. Exactness: with a deterministic system and an exact model, cpp(d) takes the
   same actions and earns the same return as paused(d) (Proposition 1);
   aligned(d) and naive(d) do not.  With an inexact model cpp(d) departs from
   paused(d), so the test is not vacuous.
3. Equal compute: aligned(d) and cpp(d) make the same number of cost calls
   on candidates of the same shape.
4. Shift invariance: adding a constant to every candidate's cost leaves the
   MPPI update unchanged.

Run: PYTHONNOUSERSITE=1 python tests/test_latency.py   (CPU, seconds; pytest also works)
"""
import pathlib
import sys

import numpy as np
import torch
try:
    import pytest
    param = pytest.mark.parametrize
except ImportError:                       # the project env has no pytest; run as a script
    param = lambda name, vals: (lambda f: setattr(f, "params", vals) or f)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from latency.schedule import MPPIConfig, PrefixMPPI, run_schedule  # noqa: E402

N, M, E = 4, 2, 3
rng = np.random.default_rng(0)
A_TRUE = torch.tensor(np.eye(N) + 0.1 * rng.standard_normal((N, N)), dtype=torch.float64)
B_TRUE = torch.tensor(0.5 * rng.standard_normal((N, M)), dtype=torch.float64)
X0 = torch.tensor(rng.standard_normal((E, N)), dtype=torch.float64)
Q, R = torch.eye(N, dtype=torch.float64), 0.1 * torch.eye(M, dtype=torch.float64)


class LinearEnv:
    def __init__(self):
        self.E, self.x = E, X0.clone()
        self.ret = torch.zeros(E, dtype=torch.float64)
        self.actions = []

    def latent(self):
        return self.x.clone()

    def step(self, u):
        u = u.clamp(-1, 1)
        self.actions.append(u.clone())
        self.ret -= (self.x @ Q * self.x).sum(-1) + (u @ R * u).sum(-1)
        self.x = self.x @ A_TRUE.T + u @ B_TRUE.T


def make_cost_factory(A_model, P, calls=None):
    def make_cost(Z):
        Zr = Z.repeat_interleave(P, 0)

        def cost(seq):
            if calls is not None:
                calls.append(tuple(seq.shape))
            x = Zr.clone(); J = torch.zeros(len(x), dtype=torch.float64)
            for t in range(seq.shape[1]):
                u = seq[:, t]
                J += (x @ Q * x).sum(-1) + (u @ R * u).sum(-1)
                x = x @ A_model.T + u @ B_TRUE.T
            return J
        return cost
    return make_cost


def run(arm, A_model=A_TRUE, H=8, T=40, calls=None):
    cfg = MPPIConfig(population=64, iterations=3, temperature=0.5)
    gen = torch.Generator().manual_seed(123)
    planner = PrefixMPPI(cfg, gen, "cpu")
    env = LinearEnv()
    run_schedule(arm, H, T, planner, make_cost_factory(A_model, cfg.population, calls), env, M)
    return env.ret, torch.stack(env.actions, 1)


@param("d", [1, 2, 4])
def test_exactness(d):
    r_p, a_p = run(f"paused{d}")
    r_c, a_c = run(f"cpp{d}")
    assert torch.allclose(a_p, a_c, atol=1e-9, rtol=0), (a_p - a_c).abs().max()
    assert torch.allclose(r_p, r_c, atol=1e-9, rtol=0)
    for other in ("aligned", "naive"):
        _, a_o = run(f"{other}{d}")
        assert (a_o - a_p).abs().max() > 1e-3, other


def test_inexact_model_departs():
    A_bad = A_TRUE + 0.05
    _, a_p = run("paused4", A_model=A_bad)
    _, a_c = run("cpp4", A_model=A_bad)
    assert (a_p - a_c).abs().max() > 1e-3


@param("d", [2, 4])
def test_equal_compute(d):
    ca, cc = [], []
    run(f"aligned{d}", calls=ca)
    run(f"cpp{d}", calls=cc)
    assert ca[1:] == cc[1:] and len(ca) == len(cc)     # [0] is the pre-episode plan (length H) in both
    assert ca[0] == cc[0]


def test_shift_invariance():
    cfg = MPPIConfig(population=32, iterations=2)
    warm = torch.zeros((2, 5, M), dtype=torch.float64)
    base = lambda s: (s ** 2).sum((1, 2))
    mu1 = PrefixMPPI(cfg, torch.Generator().manual_seed(0), "cpu")(base, warm)
    mu2 = PrefixMPPI(cfg, torch.Generator().manual_seed(0), "cpu")(lambda s: base(s) + 1234.5, warm)
    assert torch.allclose(mu1, mu2, atol=1e-10)




# --- Section 17A: hedge-then-refine (latency/refine.py) ---------------------------------------
from latency.refine import mppi_run, run_reanchor  # noqa: E402


def test_mppi_run_equals_call():
    """mppi_run from (warm, init_std) for all iterations is PrefixMPPI.__call__."""
    cfg = MPPIConfig(population=64, iterations=3, temperature=0.5)
    mk = make_cost_factory(A_TRUE, cfg.population)
    warm = torch.zeros((E, 8, M), dtype=torch.float64)
    pre = 0.3 * torch.ones((E, 2, M), dtype=torch.float64)
    p1 = PrefixMPPI(cfg, torch.Generator().manual_seed(5), "cpu")
    p2 = PrefixMPPI(cfg, torch.Generator().manual_seed(5), "cpu")
    a = p1(mk(X0), warm, pre)
    b, _ = mppi_run(p2, mk(X0), warm, torch.full_like(warm, cfg.init_std), cfg.iterations, pre)
    assert torch.allclose(a, b, atol=0, rtol=0)


def run_re(d, A_model=A_TRUE, H=8, T=40):
    cfg = MPPIConfig(population=64, iterations=3, temperature=0.5)
    planner = PrefixMPPI(cfg, torch.Generator().manual_seed(123), "cpu")
    env = LinearEnv()
    mk = make_cost_factory(A_model, cfg.population)
    run_reanchor(d, H, T, planner, mk, mk, mk, env, M)
    return env.ret, torch.stack(env.actions, 1)


@param("d", [2, 4])
def test_reanchor_exact(d):
    """With an exact model, re-anchoring on the later latent changes nothing: the re-anchored
    committed prefix reproduces committed-prefix planning (same noise stream)."""
    r_c, a_c = run(f"cpp{d}")
    r_r, a_r = run_re(d)
    assert torch.allclose(a_c, a_r, atol=1e-9), (a_c - a_r).abs().max()


def test_reanchor_inexact_departs():
    A_bad = A_TRUE + 0.05 * torch.eye(N, dtype=torch.float64)
    r_c, a_c = run("cpp4", A_model=A_bad)
    r_r, a_r = run_re(4, A_model=A_bad)
    assert not torch.allclose(a_c, a_r, atol=1e-6)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            for v in getattr(f, "params", [None]):
                (f() if v is None else f(v)); print("ok", name, "" if v is None else v)
