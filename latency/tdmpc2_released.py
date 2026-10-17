"""Study 17B (paper/docs/EXPERIMENT_PLAN_SIDE.md Section 17B): the released TD-MPC2 world models
(official single-task, state-based, 5M checkpoints) under a pipelined constant delay, the setting of
DATS, with TD-MPC2's own MPPI (envs/data_agent.py, settings unchanged).

A call starts at every decision from the current observation and supplies the action for decision
t + d.  Arms:
  paused1   the simulator waits; plan from the current latent, execute its action now (reference).
  naive<d>  plan from the current latent as if no time passed; the action executes at t + d.
  cpp<d>    roll the d pending actions through TD-MPC2's dynamics and plan from the predicted latent.
  rch<d>k<K> plan over K anchors, the prediction plus K - 1 residuals from a d-step bank in TD-MPC2's
            latent space (D_train windows of the task's collected dataset); each candidate's value
            is the mean over anchors, and the policy-prior trajectories start from the prediction.
The first d actions of every delayed episode come from the policy prior rolled through the model
from the first observation.  Episodes use seeds 950,000 + i (state observations, no rendering).
Output: results/latency/tdmpc2_released/<task>/seed<s>/<arm>.json.
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from envs.data_agent import TDMPC2Agent, checkpoint_path, load_state_dict, two_hot_inv  # noqa: E402
from envs.dmc import DMCEnv  # noqa: E402
from experiments.dmc import evaluate as EV  # noqa: E402
from paths import RESULTS  # noqa: E402

OUT = RESULTS / "latency" / "tdmpc2_released"
SEED0 = 950_000


def parse(arm):
    if arm == "paused1":
        return "paused", 0, 1
    m = re.fullmatch(r"(naive|cpp)(\d+)", arm)
    if m:
        return m.group(1), int(m.group(2)), 1
    m = re.fullmatch(r"rch(\d+)k(\d+)", arm)
    if m:
        return "rch", int(m.group(1)), int(m.group(2))
    raise ValueError(arm)


@torch.no_grad()
def rollout(ag, z, actions):
    """z (N, L), actions (N, d, A) -> latent after the d actions."""
    for t in range(actions.shape[1]):
        z = ag.dynamics(torch.cat([z, actions[:, t]], -1))
    return z


@torch.no_grad()
def plan(ag, anchors, prev_mean, gen):
    """TD-MPC2's MPPI (envs/data_agent.TDMPC2Agent._plan) from latents instead of observations.
    anchors (N, K, L); anchor 0 is the start of the policy-prior trajectories, and each candidate's
    value is the mean over the K anchors.  Returns (action (N, A), mean (H, N, A))."""
    c = ag.cfg
    H, S, E, P, A = c["horizon"], c["num_samples"], c["num_elites"], c["num_pi_trajs"], ag.act_dim
    N, K, L = anchors.shape
    z0 = anchors[:, 0]
    pi_actions = torch.empty(H, N * P, A, device=ag.device)
    _z = z0.repeat_interleave(P, dim=0)
    for t in range(H - 1):
        _, pi_actions[t] = ag._pi(_z, gen)
        _z = ag.dynamics(torch.cat([_z, pi_actions[t]], dim=-1))
    _, pi_actions[-1] = ag._pi(_z, gen)
    pi_actions = pi_actions.view(H, N, P, A)
    z = anchors.reshape(N * K, L).repeat_interleave(S, dim=0)              # (N*K*S, L), episode-major
    mean = torch.zeros(H, N, A, device=ag.device)
    std = torch.full((H, N, A), c["max_std"], device=ag.device)
    if prev_mean is not None:
        mean[:-1] = prev_mean[1:]
    actions = torch.empty(H, N, S, A, device=ag.device)
    actions[:, :, :P] = pi_actions
    for _ in range(c["iterations"]):
        r = torch.randn(H, N, S - P, A, generator=gen, device=ag.device)
        actions[:, :, P:] = (mean.unsqueeze(2) + std.unsqueeze(2) * r).clamp(-1, 1)
        aK = actions.unsqueeze(2).expand(H, N, K, S, A).reshape(H, N * K * S, A)
        value = ag._estimate_value(z, aK, gen).nan_to_num(0).view(N, K, S).mean(1)
        elite_idxs = torch.topk(value, E, dim=1).indices
        elite_value = torch.gather(value, 1, elite_idxs)
        elite_actions = torch.gather(actions, 2, elite_idxs[None, :, :, None].expand(H, N, E, A))
        max_value = elite_value.max(1, keepdim=True).values
        score = torch.exp(c["temperature"] * (elite_value - max_value))
        score = score / score.sum(1, keepdim=True)
        w, denom = score[None, :, :, None], score.sum(1)[None, :, None] + 1e-9
        mean = (w * elite_actions).sum(2) / denom
        std = ((w * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2) / denom).sqrt()
        std = std.clamp(c["min_std"], c["max_std"])
    rand_idx = torch.multinomial(score, 1, generator=gen).squeeze(1)
    a = elite_actions[0, torch.arange(N, device=ag.device), rand_idx]
    return a.clamp(-1, 1), mean


@torch.no_grad()
def residual_bank(ag, task, d, n=512, seed=11):
    """(n, L) residuals enc(o_{t+d}) - rollout(enc(o_t), a_{t:t+d}) on D_train windows (proprio)."""
    td = EV.task_data(task)
    rng = np.random.default_rng(seed + d)
    e = rng.integers(0, EV.N_TRAIN, n)
    t = rng.integers(0, td.L - d - 1, n)
    obs0 = np.stack([td.proprio[a, b] for a, b in zip(e, t)])
    obs1 = np.stack([td.proprio[a, b + d] for a, b in zip(e, t)])
    act = torch.as_tensor(np.stack([td.action[a, b:b + d] for a, b in zip(e, t)]), dtype=torch.float32, device=ag.device)
    zh = rollout(ag, ag._encode(obs0), act)
    return (ag._encode(obs1) - zh).cpu().numpy()


def run_arm(ag, task, arm, episodes, seed0, gen_seed=777, bank=None):
    kind, d, K = parse(arm)
    envs = [DMCEnv(task, render=False) for _ in range(episodes)]
    obs = np.stack([e.reset(seed0 + i)["proprio"] for i, e in enumerate(envs)])
    gen = torch.Generator(device=ag.device).manual_seed(gen_seed)
    rng = np.random.default_rng(gen_seed + 1)
    bank_t = None if bank is None else torch.as_tensor(bank, dtype=torch.float32, device=ag.device)
    ret = np.zeros(episodes)
    alive = np.ones(episodes, bool)
    prev = None
    pending = None
    if kind != "paused":                                   # first d actions: the policy prior rolled
        z = ag._encode(obs)                                # through the model from the first observation
        acts = []
        for _ in range(d):
            a, _ = ag._pi(z)
            acts.append(a)
            z = ag.dynamics(torch.cat([z, a], -1))
        pending = torch.stack(acts, 1)                     # (N, d, A)
    t_plan, n_calls, t0 = 0.0, 0, time.time()
    while alive.any():
        tp = time.perf_counter()
        z = ag._encode(obs)
        if kind in ("cpp", "rch"):
            z = rollout(ag, z, pending)
        if kind == "rch" and K > 1:
            idx = torch.as_tensor(rng.integers(0, len(bank_t), (episodes, K - 1)), device=ag.device)
            anchors = torch.cat([z[:, None], z[:, None] + bank_t[idx]], 1)
        else:
            anchors = z[:, None]
        a_new, prev = plan(ag, anchors, prev, gen)
        torch.cuda.synchronize(); t_plan += time.perf_counter() - tp; n_calls += 1
        if kind == "paused":
            a_exec = a_new
        else:
            a_exec = pending[:, 0]
            pending = torch.cat([pending[:, 1:], a_new[:, None]], 1)
        a_np = a_exec.float().cpu().numpy()
        for i, e in enumerate(envs):
            if not alive[i]:
                continue
            o, r, done = e.step(a_np[i])
            ret[i] += r; obs[i] = o["proprio"]
            if done:
                alive[i] = False
    return {"arm": arm, "d": d, "K": K, "returns": ret.tolist(), "return_mean": float(ret.mean()),
            "episodes": episodes, "seeds": [seed0, seed0 + episodes - 1], "calls": n_calls,
            "t_plan_call_mean_s": t_plan / max(n_calls, 1), "t_wall_s": time.time() - t0,
            "bank_rms": None if bank is None else float(np.sqrt((bank ** 2).sum(1).mean()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task"); ap.add_argument("seed", type=int)
    ap.add_argument("--arms", default="paused1,naive2,cpp2,rch2k4,naive4,cpp4,rch4k4,naive8,cpp8,rch8k4")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if load_state_dict(checkpoint_path(a.task, a.seed))[2]:
        sys.exit(f"{a.task}-{a.seed}: pre-release layout, excluded (planning unverified)")
    env = DMCEnv(a.task, render=False)
    od = env.reset(0)["proprio"].shape[0]
    ag = TDMPC2Agent(checkpoint_path(a.task, a.seed), od, env.act_dim, mpc=True, device="cuda")
    out = pathlib.Path(a.out) / a.task / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    banks = {}
    for arm in a.arms.split(","):
        f = out / f"{arm}.json"
        if f.exists() and not os.environ.get("FORCE"):
            print("skip (done)", f, flush=True); continue
        kind, d, K = parse(arm)
        bank = None
        if kind == "rch":
            banks.setdefault(d, residual_bank(ag, a.task, d))
            bank = banks[d]
        r = run_arm(ag, a.task, arm, a.episodes, a.seed0, bank=bank)
        r.update(task=a.task, seed=a.seed, ckpt=str(checkpoint_path(a.task, a.seed)), plan_cfg=ag.cfg,
                 gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
                 written=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        tmp = f.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(r, indent=1)); tmp.replace(f)
        print(f"tdmpc2 {a.task}-{a.seed} {arm}: return {r['return_mean']:.1f} ({r['t_wall_s']:.0f}s, "
              f"{r['t_plan_call_mean_s'] * 1e3:.1f} ms/call at batch {a.episodes})", flush=True)


if __name__ == "__main__":
    main()
