"""Ensemble and particle hedging baselines (plan Section 14, 11 October 2026).

DATS (Chen et al. 2021) hedges the post-delay state by propagating particles of
an ensemble of learned dynamics models through the committed actions and
averaging each candidate's cost over the particles.  Here the ensemble is the
M = 3 independently trained world models of a family-task cell (seeds 0, 1, 2:
the same checkpoints as every other study).  Pixel world models trained from
different seeds have different latent spaces, so a particle cannot switch
members between steps (DATS's TS-1 propagation).  Each member keeps its own
particle for the whole rollout (TS-inf):

  - every member filters its own latent from the same frames at every decision;
  - at each call, member m rolls the committed prefix through its own dynamics
    and scores every candidate's free part with its own reward head;
  - a candidate's cost is the mean over members.

Candidates are shared, so the population is that of single-model planning.
Each iteration costs M model rollouts per candidate, as RCH with K = M anchors
does.  Members are evaluated one after another, as in a direct implementation.

Arms (ensemble driver, this file):
  ens<d>        ensemble committed-prefix planning, one particle per member
  ens<d>k<K>    ensemble + RCH: each member hedges over K anchors from its own
                residual bank (exploratory)
Output: results/latency/ensemble/<family>/<task>/pop<P>_it<I>/<arm>.json

The particle baseline is a single-model arm, pcpp<d>k<K>, run by
latency/run_dmc.py; `make_cost_particles` is defined here.  It applies to the
RSSM only, the one family with a stochastic transition.  Anchor 1 is the
prediction, as in RCH; anchors 2..K are drawn from the model's own stochastic
transition through the committed actions, instead of from its residual bank.

Run in the main env with PYTHONNOUSERSITE=1 MUJOCO_GL=egl (slurm/latency_ens.sh).
"""
import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import re
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.dmc import evaluate as EV  # noqa: E402
from latency.hedge import make_cost_hedged, residual_bank  # noqa: E402
from latency.schedule import MPPIConfig, PrefixMPPI, Timer, run_schedule  # noqa: E402
from paths import RESULTS  # noqa: E402

OUT = RESULTS / "latency" / "ensemble"
SEEDS = (0, 1, 2)
STEP = {"rssm": "step0060000", "jepa": "step0060000", "tdmpc2": "step0100000"}


def member_ckpts(ckpt_root, family, task, seeds=SEEDS):
    return [pathlib.Path(ckpt_root) / "dmc" / family / task / f"seed{s}" / f"{STEP[family]}.pt" for s in seeds]


def parse(arm):
    m = re.fullmatch(r"ens(\d+)(?:k(\d+))?", arm)
    return (int(m.group(1)), int(m.group(2) or 1)) if m else None


class EnsembleBatch:
    """E episodes in lockstep through the evaluator's EnvPool; each member
    filters its own latent from the same frames."""

    def __init__(self, ads, task, seeds, workers, fake=False):
        self.ads, self.E = ads, len(seeds)
        S = EV.Settings(fake_render=fake)
        self.pool = EV.EnvPool(task, seeds, S, workers)
        self.Z = [ad.encode(self.pool.frames0) for ad in ads]
        self.ret = np.zeros(self.E)
        self.t_env = 0.0
        self.t_filter = 0.0

    def latent(self):
        return self.Z

    def step(self, act):
        a = act.float().cpu().numpy()
        t0 = time.perf_counter()
        frames, r, done = self.pool.step(a)
        self.ret += r
        t1 = time.perf_counter()
        self.t_env += t1 - t0
        self.Z = [ad.filter_step(Z, a, frames) for ad, Z in zip(self.ads, self.Z)]
        self.t_filter += time.perf_counter() - t1

    def close(self):
        self.pool.close()


def make_cost_ensemble(ads, P, dev, amp, d, H, banks=None, K=1, batch_anchors=False):
    """make_cost(Zs) for the cpp schedule: the member-averaged cost.  Member m's
    cost is RCH's (latency/hedge.py) on its own latent, with K anchors from its
    own bank; K = 1 is committed-prefix planning with that member."""
    if banks is None:
        banks = [np.zeros((1, 1))] * len(ads)          # unused at K = 1
    mcs = [make_cost_hedged(ad, P, dev, amp, d, bank, K, H, gen_seed=5 + m, batch_anchors=batch_anchors)
           for m, (ad, bank) in enumerate(zip(ads, banks))]

    def make_cost(Zs):
        costs = [mc(Z) for mc, Z in zip(mcs, Zs)]

        def cost(seq):
            return sum(c(seq) for c in costs) / len(costs)
        return cost
    return make_cost


@torch.no_grad()
def particle_anchors(ad, Z, prefix, n):
    """(n, E, D): n post-delay latents sampled from the RSSM's stochastic
    transition through the committed actions (torch's global RNG)."""
    h, z = ad._split(ad._t(Z))
    acts = ad._t(prefix)
    E = h.shape[0]
    h, z, acts = h.repeat(n, 1), z.repeat(n, 1), acts.repeat(n, 1, 1)
    for t in range(acts.shape[1]):
        h, z, _ = ad.m.img_step(h, z, acts[:, t], sample=True)
    return torch.cat([h, z], -1).reshape(n, E, -1)


def make_cost_particles(ad, P, dev, amp, d, K, H, gen_seed=5, batch_anchors=False, max_width=51_200):
    """As latency/hedge.make_cost_hedged, with anchors 2..K sampled from the
    model's stochastic transition instead of the residual bank."""
    if not hasattr(ad.m, "img_step"):
        raise ValueError("particle anchors need a stochastic transition (RSSM)")
    torch.manual_seed(gen_seed)

    def ac():
        return torch.autocast("cuda", dtype=torch.float16) if amp else torch.autocast("cpu", enabled=False)

    def make_cost(Z):
        Z = np.asarray(Z)
        E = len(Z)

        def cost(seq):
            L, A = seq.shape[1], seq.shape[2]
            if L == H:                                                  # pre-episode plan: state known
                Zr = torch.as_tensor(Z, dtype=torch.float32, device=dev).repeat_interleave(P, 0)
                with ac():
                    return torch.as_tensor(ad.get_cost(Zr, seq.float()), dtype=torch.float64, device=dev)
            assert L == H + d
            s = seq.reshape(E, P, L, A)
            prefix = s[:, 0, :d].float().cpu().numpy()
            zh = torch.as_tensor(np.asarray(ad.rollout(Z, prefix))[:, -1], dtype=torch.float32, device=dev)
            anc = [zh] + list(particle_anchors(ad, Z, prefix, K - 1)) if K > 1 else [zh]
            free = s[:, :, d:].float().reshape(E * P, L - d, A)
            if batch_anchors:
                Zall = torch.stack(anc, 0).repeat_interleave(P, 1).reshape(K * E * P, -1)
                Fall = free.unsqueeze(0).expand(K, *free.shape).reshape(K * E * P, *free.shape[1:])
                Js = []
                for i in range(0, len(Zall), max_width):
                    with ac():
                        Js.append(torch.as_tensor(ad.get_cost(Zall[i:i + max_width], Fall[i:i + max_width]),
                                                  dtype=torch.float64, device=dev))
                return torch.cat(Js).reshape(K, E * P).mean(0)
            J = torch.zeros(E * P, dtype=torch.float64, device=dev)
            for a_k in anc:
                with ac():
                    J += torch.as_tensor(ad.get_cost(a_k.repeat_interleave(P, 0), free), dtype=torch.float64,
                                         device=dev)
            return J / K
        return cost
    return make_cost


def run_arm(ads, metas, task, arm, cfg, episodes, workers, dev, amp, T, fake, H, planner_seed=777, seed0=950_000):
    d, K = parse(arm)
    A = EV.task_data(task).act_dim
    seeds = list(range(seed0, seed0 + episodes))
    banks = [residual_bank(ad, task, d) for ad in ads] if K > 1 else None
    make_cost = make_cost_ensemble(ads, cfg.population, dev, amp, d, H, banks, K)
    env = EnsembleBatch(ads, task, seeds, workers, fake)
    planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(planner_seed), dev)
    timer = Timer(dev)
    t0 = time.time()
    try:
        n_calls = run_schedule(f"cpp{d}", H, T, planner, make_cost, env, A, timer)
    finally:
        env.close()
    pc = np.asarray(timer.per_call)
    out = {"arm": arm, "M": len(ads), "K": K, "returns": env.ret.tolist(), "return_mean": float(env.ret.mean()),
           "episodes": episodes, "seeds": [seeds[0], seeds[-1]], "decisions": T, "H": H, "n_calls": n_calls,
           "t_plan_s": float(pc.sum()), "t_plan_call_mean_s": float(pc.mean()),
           "t_plan_call_p95_s": float(np.percentile(pc, 95)), "t_env_s": env.t_env, "t_filter_s": env.t_filter,
           "t_wall_s": time.time() - t0, "nonfinite_frac": planner.n_bad / max(planner.n_all, 1),
           "batch_size": episodes, "members": [m["run"] for m in metas]}
    if banks is not None:
        out["bank_rms"] = [float(np.sqrt((b ** 2).sum(1).mean())) for b in banks]
    return out


def main():
    from latency.run_dmc import H, git_hash, load_adapter
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("family", choices=sorted(STEP))
    ap.add_argument("task")
    ap.add_argument("--arms", required=True, help="comma-separated, e.g. ens4,ens8,ens4k4")
    ap.add_argument("--ckpt-root", default=os.environ.get("CKPT_KEEP",
                                                          "/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep"))
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=950_000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("DMC_ENV_WORKERS", 0)))
    ap.add_argument("--fake-render", action="store_true", help="smoke test without EGL")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    if amp:
        EV.check_precision(autocast_dtype=torch.float16)
    cks = member_ckpts(a.ckpt_root, a.family, a.task)
    loaded = [load_adapter(c, dev) for c in cks]
    ads, metas = [x[0] for x in loaded], [x[1] for x in loaded]
    assert all(m["family"] == a.family and m["task"] == a.task for m in metas)
    cfg = MPPIConfig(population=a.pop, iterations=a.iters)
    od = pathlib.Path(a.out) / a.family / a.task / f"pop{a.pop}_it{a.iters}"
    od.mkdir(parents=True, exist_ok=True)
    failed = []
    for arm in a.arms.split(","):
        f = od / f"{arm}.json"
        if f.exists() and not os.environ.get("FORCE"):
            print("skip (done)", f, flush=True); continue
        try:
            r = run_arm(ads, metas, a.task, arm, cfg, a.episodes, a.workers, dev, amp, a.T, a.fake_render, H,
                        seed0=a.seed0)
        except Exception:
            traceback.print_exc(); failed.append(arm); continue
        r.update(metas=metas, ckpts=[str(c) for c in cks], config=dataclasses.asdict(cfg),
                 gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", git=git_hash(),
                 host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
                 written=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        tmp = f.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(r, indent=1)); tmp.replace(f)
        print(f"{a.family}/{a.task} ensemble pop{a.pop} it{a.iters} {arm}: return {r['return_mean']:.1f} "
              f"({r['n_calls']} calls, {r['t_plan_call_mean_s'] * 1e3:.0f} ms/call at batch {a.episodes}, "
              f"{r['t_wall_s']:.0f}s)", flush=True)
    if failed:
        sys.exit(f"failed arms: {failed}")


if __name__ == "__main__":
    main()
