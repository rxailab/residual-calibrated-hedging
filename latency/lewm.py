"""Study 15B (paper/docs/EXPERIMENT_PLAN_SIDE.md Section 15): delay schedules and
residual-calibrated hedging with the public LeWM world models (Maes et al. 2026;
quentinll/lewm-{pusht,cube,reacher}), run inside stable-worldmodel's own dataset-driven
evaluation (swm.World.evaluate), so LeWM's start and goal states, budget and success
criterion are unchanged.

A decision is one LeWM model step: one 10-dimensional action block, executed as
action_block = 5 environment steps.  The planner is LeWM's CEM (300 samples, 30
iterations, 30 elites, initial variance 1, first sample the mean, no clamping in the
normalised action space), batched over environments, with horizon H = 5 blocks and a
warm start shifted by the blocks executed.  The cost is LeWM's: the squared distance
between the predicted embedding at the end of the candidate and the goal embedding.

Each plan is executed for r = 5 blocks (--period; LeWM's receding horizon, which executes the
whole optimised sequence), so every schedule replans every r decisions.  Arms:
  paused<p>   the simulator waits for each call; a new plan every p decisions (paused5 is
              LeWM's own evaluation).
  aligned<d>  calls take d decisions: the next call is issued d decisions before the current
              plan runs out, from the observation at that decision, with candidates d + H
              long; blocks d:d+r are executed after the call.
  naive<d>    the same plan, blocks 0:r executed.
  cpp<d>      committed prefix: the d blocks that execute during the call are fixed in every
              candidate, [committed d blocks, free H blocks]; the free blocks 0:r are executed.
  rch<d>k<K>  cpp with residual-calibrated hedging: the embedding after the prefix is
              replaced by K anchors, the prediction plus K - 1 residuals drawn from a bank
              of LeWM's own d-step errors on dataset windows, and a candidate's cost is the
              mean over anchors (all anchors in one batched call).
The first plan is made with the simulator paused (the robot at rest before the episode).
With r = d these are the schedules of latency/schedule.py.  Plan Section 15B, amendment of
16 October 2026: replanning every decision, as first planned, defeats LeWM's terminal-only
cost (paused1 succeeds 0.32 against 0.98 for paused5 on LeWM's own protocol;
latency/lewm_validate.py).

Residual bank: 512 windows of the task's dataset, from episodes not used for evaluation;
r = enc(o_{t+5d}) - rollout(z_t, a_{t:t+5d}), with the recorded actions normalised and
blocked as in LeWM's training.

Output: results/latency/lewm/r<period>_g<goal offset>_b<budget>/<task>/<arm>.json.
Run with the rch-lewm env: STABLEWM_HOME=/scratch/hpc/11/xiar3/lewm MUJOCO_GL=egl.
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = pathlib.Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402

LEWM = pathlib.Path(os.environ.get("STABLEWM_HOME", "/scratch/hpc/11/xiar3/lewm"))
OUT = ROOT / "results" / "latency" / "lewm"
SEED0 = 950_000
H, BLOCK, GOAL_OFFSET, BUDGET = 5, 5, 25, 50

# LeWM's evaluation configs (le-wm/config/eval/*.yaml at 8edfeb3), unchanged
TASKS = {
    "pusht": dict(world=dict(env_name="swm/PushT-v1"), dataset="pusht_expert_train", ckpt="hf_pusht",
                  keys=["action", "proprio", "state"],
                  callables=[{"method": "_set_state", "args": {"state": {"value": "state"}}},
                             {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}}]),
    "reacher": dict(world=dict(env_name="swm/ReacherDMControl-v0", task="qpos_match"), dataset="dmc/reacher_random",
                    ckpt="hf_reacher", keys=["action"],
                    callables=[{"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
                               {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}}]),
    "cube": dict(world=dict(env_name="swm/OGBCube-v0", env_type="single", ob_type="states", multiview=False,
                            width=224, height=224, visualize_info=False, terminate_at_goal=True),
                 dataset="ogbench/cube_single_expert", ckpt="hf_cube", keys=["action"],
                 callables=[{"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
                            {"method": "set_target_pos", "args": {
                                "cube_id": {"value": 0, "in_dataset": False},
                                "target_pos": {"value": "goal_privileged_block_0_pos"},
                                "target_quat": {"value": "goal_privileged_block_0_quat"}}}]),
}
CEM = dict(num_samples=300, n_steps=30, topk=30, var_scale=1.0)


def img_transform():
    return transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                               transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size=224)])


def parse(arm):
    m = re.fullmatch(r"(paused|aligned|cpp|naive)(\d+)", arm)
    if m:
        return m.group(1), int(m.group(2)), 1
    m = re.fullmatch(r"rch(\d+)k(\d+)", arm)
    if m:
        return "rch", int(m.group(1)), int(m.group(2))
    raise ValueError(f"bad arm {arm!r}")


# ------------------------------------------------------------------ model
@torch.inference_mode()
def rollout(model, emb_hist, act_seq):
    """stable_worldmodel.wm.lewm.LeWM.rollout on embeddings: emb_hist (N, h, D) are the
    history frames, act_seq (N, L, 10) the actions, the first h aligned with the history
    frames.  Predicts L - h + 1 further embeddings; returns all (N, L + 1, D)."""
    h = emb_hist.shape[1]
    HS = model.predictor.num_frames
    act_emb = model.action_encoder(act_seq)
    embs = list(emb_hist.unbind(1))
    for t in range(act_seq.shape[1] - h + 1):
        lo = max(0, h + t - HS)
        embs.append(model.predict(torch.stack(embs[lo:], 1), act_emb[:, lo:h + t])[:, -1])
    return torch.stack(embs, 1)


@torch.inference_mode()
def encode(model, pixels):
    """pixels: (N, 3, 224, 224) transformed -> (N, D)."""
    return model.encode({"pixels": pixels[:, None]})["emb"][:, 0]


def cost_fn(model, z0, goal, prefix=None, anchors=None):
    """Returns cost(cand) for candidates (E, S, L, 10) of each environment.
    z0, goal: (E, D).  prefix: (E, d, 10) committed blocks or None.  anchors: (E, K, D)
    offsets added to the embedding after the prefix (RCH; the first is zero)."""
    E, D = z0.shape

    if prefix is None:
        def cost(cand):
            S, L = cand.shape[1], cand.shape[2]
            z = rollout(model, z0[:, None, None].expand(E, S, 1, D).reshape(E * S, 1, D), cand.reshape(E * S, L, -1))
            return ((z[:, -1] - goal.repeat_interleave(S, 0)) ** 2).sum(-1).reshape(E, S)
        return cost

    d = prefix.shape[1]
    hist = rollout(model, z0[:, None], prefix)                       # (E, d + 1, D): z_t, z^_{t+1..t+d}
    K = 1 if anchors is None else anchors.shape[1]
    if anchors is not None:
        hist = hist[:, None].expand(E, K, d + 1, D).clone()
        hist[:, :, -1] += anchors
    else:
        hist = hist[:, None]

    def cost(cand):
        S, L = cand.shape[1], cand.shape[2]
        acts = torch.cat([prefix[:, None].expand(E, S, d, prefix.shape[-1]), cand], 2)       # (E, S, d + H, 10)
        acts = acts[:, None].expand(E, K, S, d + L, acts.shape[-1]).reshape(E * K * S, d + L, -1)
        hh = hist[:, :, None].expand(E, K, S, d + 1, D).reshape(E * K * S, d + 1, D)
        z = rollout(model, hh, acts)
        g = goal[:, None, None].expand(E, K, S, D).reshape(E * K * S, D)
        return ((z[:, -1] - g) ** 2).sum(-1).reshape(E, K, S).mean(1)
    return cost


@torch.inference_mode()
def cem(cost, mean, gen, dev):
    """stable_worldmodel.solver.CEMSolver.solve, batched over environments.  mean (E, L, 10)."""
    var = torch.full_like(mean, CEM["var_scale"])
    for _ in range(CEM["n_steps"]):
        cand = torch.randn((mean.shape[0], CEM["num_samples"], *mean.shape[1:]), generator=gen, device=dev)
        cand = cand * var[:, None] + mean[:, None]
        cand[:, 0] = mean
        c = cost(cand)
        idx = torch.topk(c, k=CEM["topk"], dim=1, largest=False).indices
        elite = cand[torch.arange(cand.shape[0], device=dev)[:, None], idx]
        mean, var = elite.mean(1), elite.std(1)
    return mean


# ------------------------------------------------------------------ policy
def shift(mu, k):
    """Warm start for a plan that starts k blocks later (latency/schedule.py)."""
    k = min(k, mu.shape[1])
    return torch.cat([mu[:, k:], torch.zeros_like(mu[:, :k])], 1)


class DelayPolicy(swm.policy.BasePolicy):
    """Delay schedules over decisions of BLOCK environment steps, all environments in lockstep."""

    def __init__(self, model, arm, process, transform, bank=None, seed=777, dev="cuda", period=5, handover=None):
        super().__init__()
        self.r = period
        # handover: (E, d, 10) normalised blocks of recorded actions that execute during the first call
        self.handover = None if handover is None else torch.as_tensor(handover, dtype=torch.float32, device=dev)
        self.type = "world_model"
        self.model, self.dev = model, dev
        self.kind, self.d, self.K = parse(arm)
        self.process, self.transform = process, transform
        self.bank = None if bank is None else torch.as_tensor(bank, dtype=torch.float32, device=dev)
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.rng = np.random.default_rng(seed + 1)
        self.calls, self.call_s = 0, []

    def set_env(self, env):
        self.env = env
        self.n = env.num_envs
        self.adim = env.single_action_space.shape[-1]
        self.step = 0
        self.queue = []                        # blocks (E, 10) for the coming decisions
        self.pending = None                    # the plan of the call in flight
        self.warm = torch.zeros((self.n, H, self.adim * BLOCK), device=self.dev)

    def _plan(self, info, prefix=None, length=H):
        t0 = time.perf_counter()
        pix = info["pixels"][:, -1].to(self.dev)
        goal = info["goal"][:, -1].to(self.dev)
        z0, g = encode(self.model, pix), encode(self.model, goal)
        anchors = None
        if self.kind == "rch" and prefix is not None and self.K > 1:
            idx = torch.as_tensor(self.rng.integers(0, len(self.bank), (self.n, self.K - 1)), device=self.dev)
            anchors = torch.cat([torch.zeros_like(z0)[:, None], self.bank[idx]], 1)
        mean = self.warm[:, :length]
        if mean.shape[1] < length:
            mean = torch.cat([mean, torch.zeros((self.n, length - mean.shape[1], mean.shape[2]), device=self.dev)], 1)
        mu = cem(cost_fn(self.model, z0, g, prefix, anchors), mean, self.gen, self.dev)
        torch.cuda.synchronize()
        self.call_s.append(time.perf_counter() - t0); self.calls += 1
        return mu

    def _decide(self, info):
        """Schedules with plans of r = self.r blocks (LeWM's receding horizon) and calls of d
        decisions.  paused<p>: the simulator waits, a new plan every p decisions.  Asynchronous:
        the next call is issued d decisions before the current plan runs out, from the
        observation at that decision; cpp and rch take the d remaining blocks as the committed
        prefix and return a plan of H free blocks, of which r are executed; aligned plans d + H
        blocks from the observation and executes blocks d:d+r; naive executes blocks 0:r.  With
        r = d this is latency/schedule.py.  The first plan is made with the simulator paused."""
        k, d, r = self.step // BLOCK, self.d, self.r
        if k == 0 and self.handover is not None:                     # moving start: the system is executing
            self.queue = list(self.handover.unbind(1))               # recorded actions when the planner starts
            self.pending = None
        if self.kind == "paused":
            if not self.queue:
                mu = self._plan(info)
                self.queue = list(mu[:, :d].unbind(1))
                self.warm = shift(mu, d)
            return
        if k == 0 and self.handover is None:                         # pre-episode plan, simulator paused
            mu0 = self._plan(info)
            self.queue = list(mu0[:, :r].unbind(1))
            self.pending = None
            # the next call's first optimised block: plan start (cpp, rch) or issue time (aligned, naive)
            self.warm = shift(mu0, r) if self.kind in ("cpp", "rch") else shift(mu0, r - d)
        if not self.queue and self.pending is not None:              # the call issued d decisions ago returns
            self.queue, self.pending = list(self.pending.unbind(1)), None
        if len(self.queue) == d and self.pending is None:            # issue the next call
            if self.kind in ("cpp", "rch"):
                mu = self._plan(info, prefix=torch.stack(self.queue, 1))
                self.pending = mu[:, :r]
            else:
                mu = self._plan(info, length=d + H)
                self.pending = mu[:, d:d + r] if self.kind == "aligned" else mu[:, :r]
            self.warm = shift(mu, r)

    def get_action(self, info_dict, **kwargs):
        info = self._prepare_info(info_dict)
        if self.step % BLOCK == 0:
            self._decide(info)
            self.block = self.queue.pop(0)
        j = self.step % BLOCK
        a = self.block[:, j * self.adim:(j + 1) * self.adim].float().cpu().numpy()
        self.step += 1
        a = a.reshape(*self.env.action_space.shape)
        return self.process["action"].inverse_transform(a) if "action" in self.process else a


# ------------------------------------------------------------------ data
def load_dataset(task):
    c = TASKS[task]
    return swm.data.HDF5Dataset(c["dataset"], keys_to_cache=c["keys"], cache_dir=LEWM)


def processors(dataset, keys):
    proc = {}
    for col in keys:
        x = dataset.get_col_data(col)
        x = x[~np.isnan(x).any(axis=1)]
        proc[col] = preprocessing.StandardScaler().fit(x)
        if col != "action":
            proc[f"goal_{col}"] = proc[col]
    return proc


def episode_lengths(dataset):
    """Per-row episode index, step index, and length (max step + 1) of each episode id."""
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = np.asarray(dataset.get_col_data(col)).astype(np.int64)
    step = np.asarray(dataset.get_col_data("step_idx")).astype(np.int64)
    L = np.zeros(ep.max() + 1, np.int64)
    np.maximum.at(L, ep, step)
    return col, ep, step, L + 1


def eval_starts(dataset, n, seed, goal_offset=GOAL_OFFSET):
    """LeWM's eval.py sampling of start rows, with our seed (same rule, vectorised)."""
    col, ep, step, L = episode_lengths(dataset)
    max_start = L[ep] - goal_offset - 1
    valid = np.nonzero(step <= max_start)[0]
    g = np.random.default_rng(seed)
    rows = np.sort(valid[g.choice(len(valid) - 1, size=n, replace=False)])
    data = dataset.get_row_data(rows)
    return data[col], data["step_idx"]


@torch.inference_mode()
def residual_bank(model, dataset, task, d, exclude, process, dev, n=512, seed=11):
    col, ep, step, Lall = episode_lengths(dataset)
    ids = np.setdiff1d(np.unique(ep), np.asarray(exclude))
    rng = np.random.default_rng(seed + d)
    tf = img_transform()
    out = []
    while sum(len(o) for o in out) < n:
        e = rng.choice(ids, 64)
        lens = Lall[e]
        ok = lens > BLOCK * d + 2
        e, lens = e[ok], lens[ok]
        t = np.array([rng.integers(0, L - BLOCK * d - 1) for L in lens])
        chunks = dataset.load_chunk(e, t, t + BLOCK * d + 1)
        acts, p0, p1 = [], [], []
        for ch in chunks:
            a = ch["action"][:BLOCK * d]
            a = a.numpy() if torch.is_tensor(a) else np.asarray(a)
            if np.isnan(a).any():
                continue
            acts.append(process["action"].transform(a).reshape(d, -1))
            px = ch["pixels"]                                       # (T, 3, H, W) uint8 from load_chunk
            p0.append(tf(px[0])); p1.append(tf(px[-1]))
        if not acts:
            continue
        z0 = encode(model, torch.stack(p0).to(dev))
        z1 = encode(model, torch.stack(p1).to(dev))
        zh = rollout(model, z0[:, None], torch.as_tensor(np.stack(acts), dtype=torch.float32, device=dev))[:, -1]
        out.append((z1 - zh).cpu().numpy())
    return np.concatenate(out)[:n]


def handover_blocks(dataset, eps, starts, d, process):
    """(E, d, 10): the recorded actions of each evaluation episode from its start step, normalised
    and blocked as LeWM's actions."""
    chunks = dataset.load_chunk(np.asarray(eps), np.asarray(starts), np.asarray(starts) + BLOCK * d)
    out = []
    for ch in chunks:
        a = ch["action"][:BLOCK * d]
        a = a.numpy() if torch.is_tensor(a) else np.asarray(a)
        assert not np.isnan(a).any(), "hand-over actions must lie inside the episode"
        out.append(process["action"].transform(a).reshape(d, -1))
    return np.stack(out)


def load_model(task, dev):
    m = swm.wm.utils.load_pretrained(str(LEWM / TASKS[task]["ckpt"]))
    m = m.to(dev).eval()
    m.requires_grad_(False)
    m.interpolate_pos_encoding = True
    return m


def git_hash():
    try:
        return subprocess.run(["/usr/shared_apps/packages/git-2.49.0/bin/git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=50, help="environments per World")
    ap.add_argument("--seed", type=int, default=SEED0)
    ap.add_argument("--period", type=int, default=5, help="blocks per plan (LeWM's receding horizon)")
    ap.add_argument("--goal-offset", type=int, default=GOAL_OFFSET)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--handover", type=int, default=0,
                    help="moving start: the first N decisions execute the recorded actions (Section 15B, amendment 2)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    dev = "cuda"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    c = TASKS[a.task]
    model = load_model(a.task, dev)
    dataset = load_dataset(a.task)
    process = processors(dataset, c["keys"])
    eps, starts = eval_starts(dataset, a.episodes, a.seed, a.goal_offset)
    od = pathlib.Path(a.out) / (f"r{a.period}_g{a.goal_offset}_b{a.budget}" + (f"_h{a.handover}" if a.handover else "")) / a.task
    od.mkdir(parents=True, exist_ok=True)
    banks = {}
    for arm in a.arms.split(","):
        f = od / f"{arm}.json"
        if f.exists() and not os.environ.get("FORCE"):
            print("skip (done)", f, flush=True); continue
        kind, d, K = parse(arm)
        bank = None
        if kind == "rch":
            if d not in banks:
                banks[d] = residual_bank(model, dataset, a.task, d, np.unique(eps), process, dev)
            bank = banks[d]
        succ, call_s, t0 = [], [], time.time()
        for i in range(0, a.episodes, a.chunk):
            sl = slice(i, i + a.chunk)
            n = len(eps[sl])
            ho = None
            if a.handover:
                if kind != "paused" and d != a.handover:
                    raise ValueError("the hand-over lasts one call: --handover must equal the arm's delay")
                ho = handover_blocks(dataset, eps[sl], starts[sl], a.handover, process)
            world = swm.World(**c["world"], num_envs=n, max_episode_steps=2 * a.budget, image_shape=(224, 224))
            pol = DelayPolicy(model, arm, process, {"pixels": img_transform(), "goal": img_transform()}, bank,
                              seed=777 + i, dev=dev, period=a.period, handover=ho)
            world.set_policy(pol)
            m = world.evaluate(dataset=dataset, start_steps=list(map(int, starts[sl])), goal_offset=a.goal_offset,
                               eval_budget=a.budget, episodes_idx=list(map(int, eps[sl])), callables=c["callables"])
            succ += np.asarray(m["episode_successes"], bool).tolist()
            call_s += pol.call_s
            world.close()
        r = {"task": a.task, "arm": arm, "d": d, "K": K, "successes": succ, "success_rate": float(np.mean(succ)),
             "episodes": len(succ), "episode_idx": list(map(int, eps)), "start_steps": list(map(int, starts)),
             "seed": a.seed, "H": H, "block": BLOCK, "period": a.period, "handover": a.handover,
             "goal_offset": a.goal_offset,
             "budget": a.budget, "cem": CEM,
             "calls": len(call_s), "call_s_mean_batch": float(np.mean(call_s)), "chunk": a.chunk,
             "bank_n": None if bank is None else len(bank),
             "bank_rms": None if bank is None else float(np.sqrt((bank ** 2).sum(1).mean())),
             "t_wall_s": time.time() - t0, "gpu": torch.cuda.get_device_name(0), "git": git_hash(),
             "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"),
             "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        tmp = f.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(r, indent=1)); tmp.replace(f)
        print(f"lewm {a.task} {arm}: success {r['success_rate']:.3f} ({len(succ)} episodes, {r['t_wall_s']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
