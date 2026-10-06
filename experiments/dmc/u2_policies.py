"""U2 policy set for DMControl (plan Section 5, U2): 12 state-based policies
built on the task's TD-MPC2 policy prior, their true returns, and the common
random numbers that make imagined and true rollouts paired.

Policies (index: kind, parameter)
  0-7   "eps"    the prior's deterministic action tanh(mu); with probability
                 eps it is replaced by a uniform random action,
                 eps in {0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 0.9}
  8-10  "gauss"  the prior's action plus N(0, sigma^2) noise, clipped,
                 sigma in {0.2, 0.5, 1.0}
  11    "random" uniform random actions in [-1, 1]
The eps ladder stops at 0.9 because eps = 1 is the uniform random policy,
which would put the same policy in the set twice.

The prior is the released data agent (experiments/dmc/collect.py AGENT_SEED)
without planning (mpc=False), for every task.  Each (policy, start seed) pair
owns a random stream drawn up front (seed 10000 * policy + start seed, as in
the LunarLander evaluator), so a policy makes the same random choices inside
every world model and in the real environment.

True returns are the mean over 50 fixed start seeds of the 100-decision
return in the real environment, computed once per task on the CPU and cached
in artifacts/dmc/u2_true_<task>.json.  The start frames (rendered at 128x128
and pooled exactly like the training data) are cached beside them in
artifacts/dmc/u2_start_<task>.npy; rendering needs a GPU node.
"""
from __future__ import annotations
import sys, pathlib, json, os, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

POLICIES = ([("eps", e) for e in (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 0.9)] +
            [("gauss", s) for s in (0.2, 0.5, 1.0)] + [("random", 0.0)])
U2_SEEDS = list(range(800_000, 800_050))        # S_U2, shared by every checkpoint
U2_H = 100                                      # H_eval, plan Appendix A
CACHE = ROOT / "artifacts" / "dmc"


def pool(px):
    """A 128x128 render -> the 64x64 frame the models were trained on, by the
    exact rule of experiments/dmc/preprocess.py: mean over 2x2 blocks, rint,
    uint8."""
    px = np.asarray(px)
    small = px.reshape(*px.shape[:-3], 64, 2, 64, 2, 3).mean(axis=(-4, -2))
    return np.rint(small).astype(np.uint8)


def crn_draws(act_dim, seeds=U2_SEEDS, H=U2_H):
    """Per-row random numbers for all 12 x len(seeds) (policy, seed) rows,
    policy-major.  Drawn up front so the draws never depend on the states."""
    n = len(POLICIES) * len(seeds)
    u, unif, gauss = np.empty((n, H)), np.empty((n, H, act_dim)), np.empty((n, H, act_dim))
    kind, param = [], []
    for pi, (k, p) in enumerate(POLICIES):
        for j, s in enumerate(seeds):
            rng = np.random.default_rng(10_000 * pi + s)
            row = pi * len(seeds) + j
            u[row] = rng.random(H)
            unif[row] = rng.uniform(-1, 1, (H, act_dim))
            gauss[row] = rng.standard_normal((H, act_dim))
            kind.append(k); param.append(p)
    return {"u": u, "unif": unif, "gauss": gauss, "kind": np.asarray(kind),
            "param": np.asarray(param, float)}


def policy_actions(base, dr, t, rows=slice(None)):
    """Actions of the policy rows `rows` at decision t, given the prior's
    deterministic actions `base` for those rows."""
    base = np.asarray(base, np.float64)
    kind, param = dr["kind"][rows], dr["param"][rows]
    u, unif, gauss = dr["u"][rows, t], dr["unif"][rows, t], dr["gauss"][rows, t]
    a = base.copy()
    swap = (kind == "eps") & (u < param)
    a[swap] = unif[swap]
    g = kind == "gauss"
    a[g] = base[g] + param[g, None] * gauss[g]
    r = kind == "random"
    a[r] = unif[r]
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def load_prior(task, device="cpu"):
    from envs.data_agent import TDMPC2Agent, checkpoint_path
    from experiments.dmc.collect import AGENT_SEED
    from envs.dmc import DMCEnv
    env = DMCEnv(task, render=False)
    obs_dim = env.reset(0)["proprio"].shape[0]
    return TDMPC2Agent(checkpoint_path(task, AGENT_SEED[task]), obs_dim, env.act_dim,
                       mpc=False, device=device)


def _atomic_json(f, obj):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj)); tmp.replace(f)


def true_returns(task, seeds=U2_SEEDS, H=U2_H, cache=True):
    """(12,) mean true H-decision returns, and the (12, len(seeds)) per-seed
    returns.  Computed once on the CPU and cached."""
    f = CACHE / f"u2_true_{task}.json"
    if cache and f.exists():
        d = json.load(open(f))
        if d["seeds"] == list(seeds) and d["H"] == H and d["policies"] == [list(p) for p in POLICIES]:
            return np.asarray(d["mean"]), np.asarray(d["returns"])
    from envs.dmc import DMCEnv
    t0 = time.time()
    prior = load_prior(task, "cpu")
    n = len(seeds)
    envs = [DMCEnv(task, render=False) for _ in seeds]
    envs[0].reset(seeds[0])                       # act_dim is known once the physics is loaded
    dr = crn_draws(envs[0].act_dim, seeds, H)
    R = np.zeros((len(POLICIES), n))
    for pi in range(len(POLICIES)):
        rows = slice(pi * n, (pi + 1) * n)
        prop = np.stack([e.reset(s)["proprio"] for e, s in zip(envs, seeds)])
        for t in range(H):
            a = policy_actions(prior.act(prop), dr, t, rows)
            for k, e in enumerate(envs):
                o, r, done = e.step(a[k]); R[pi, k] += r; prop[k] = o["proprio"]
                assert not done, "U2 horizon must end before the time limit"
    if cache:
        _atomic_json(f, {"task": task, "policies": [list(p) for p in POLICIES], "seeds": list(seeds),
                         "H": H, "prior": "TD-MPC2 5M policy prior, mpc=False, CPU",
                         "mean": R.mean(1).tolist(), "returns": R.tolist(),
                         "seconds": round(time.time() - t0, 1)})
    return R.mean(1), R


def start_frames(task, seeds=U2_SEEDS, cache=True, fake=False):
    """(len(seeds), 64, 64, 3) uint8 first frames of the U2 start states."""
    if fake:                      # CPU smoke tests only: the login node cannot render
        return np.zeros((len(seeds), 64, 64, 3), np.uint8)
    f = CACHE / f"u2_start_{task}.npy"
    if cache and f.exists():
        fr = np.load(f)
        if len(fr) == len(seeds):
            return fr
    from envs.dmc import DMCEnv
    env = DMCEnv(task, img=128)
    fr = np.stack([pool(env.reset(s)["pixels"]) for s in seeds])
    if cache:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npy")
        np.save(tmp, fr); tmp.replace(f)
    return fr


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="build the U2 true-return caches")
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--frames", action="store_true", help="also render the start frames (GPU node)")
    a = ap.parse_args()
    for task in a.tasks:
        m, R = true_returns(task)
        print(task, "true returns:", np.round(m, 1).tolist(), flush=True)
        if a.frames:
            print(task, "start frames:", start_frames(task).shape, flush=True)
