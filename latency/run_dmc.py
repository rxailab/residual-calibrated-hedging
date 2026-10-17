"""Studies 1 and 2 of the side paper: delay-in-decisions closed loop for the
RSSM, JEPA (3-frame history) and TD-MPC2 DMControl models.

One invocation = one model (checkpoint) and one planner configuration; it runs
every arm of the chosen preset and writes one JSON per arm, skipping arms that
already have a result:

  results/latency/dmc/<family>/<task>/<run>/pop<P>_it<I>/<arm>.json

Arms beyond the schedules (latency/schedule.py): hcpp<d>k<K> (RCH, latency/hedge.py),
the delay-error arms (latency/delay_error.py), pcpp<d>k<K> (particle anchors from the
RSSM's stochastic transition, latency/ensemble.py) and rcpp<d>k<K>r<m> (hedge-then-refine: the
last m MPPI iterations re-anchored on a fresher latent, latency/refine.py).

Presets
  study1  paused1; paused, naive, aligned, cpp at d in {2, 4, 8, 16}; naive,
          aligned, cpp at d = 1 (paused1 is paused at d = 1): 20 arms.
  study2  paused1 and cpp at d in {1, 2, 4, 8, 16}: 6 arms.

Episodes use initial-state seeds 950,000 + i (disjoint from the main study's
S_U1).  Every arm of a model uses the same seeds and the same planner noise
stream (common random numbers).  Latents are filtered at every decision with
the model's own filter_step, whatever the planning schedule.

Run in the main env with PYTHONNOUSERSITE=1 MUJOCO_GL=egl (slurm/latency_dmc.sh).
"""
import argparse
import dataclasses
import datetime
import json
import os
import re
import pathlib
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.dmc import evaluate as EV  # noqa: E402  (imported, never modified)
from latency.schedule import MPPIConfig, PrefixMPPI, Timer, run_schedule  # noqa: E402
from latency.hedge import make_cost_hedged, residual_bank  # noqa: E402
from latency.delay_error import fit_delay_error, make_cost_delay_error, parse as parse_de  # noqa: E402
from paths import RESULTS  # noqa: E402

OUT = RESULTS / "latency" / "dmc"
SEED0 = 950_000
H = 32                                   # H_plan, amendment A10
PRESETS = {
    "study1": ["paused1"] + [f"{k}1" for k in ("naive", "aligned", "cpp")]
              + [f"{k}{d}" for d in (2, 4, 8, 16) for k in ("paused", "naive", "aligned", "cpp")],
    "study2": ["paused1"] + [f"cpp{d}" for d in (1, 2, 4, 8, 16)],
}


def git_hash():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                               text=True, env={**os.environ, "PATH": "/usr/shared_apps/packages/git-2.49.0/bin:"
                                               + os.environ.get("PATH", "")}).stdout.strip()
    except Exception:
        return None


def load_adapter(ckpt, dev):
    """The evaluator's construction (experiments/dmc/evaluate.py, evaluate()), without the probe."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    family = ck["family"]
    hist = family == "jepa" and ck.get("arch") == "hist3"
    if family == "tdmpc2":
        model = EV.PixelTDMPC2(act_dim=ck["act_dim"], size=ck["size"])
        model.load_state_dict(ck["state_dict"])
        ad = EV.as_torch_input(EV.TDMPC2Adapter(model, dev))
    else:
        model = (EV.PixelJEPAHist if hist else {"rssm": EV.PixelRSSM, "jepa": EV.PixelJEPA}[family])(
            act_dim=ck["act_dim"], width=ck["width"])
        model.load_state_dict(ck["state_dict"])
        ad = EV.as_torch_input((EV.PixelJEPAHistAdapter if hist else
                                {"rssm": EV.PixelRSSMAdapter, "jepa": EV.PixelJEPAAdapter}[family])(model, dev))
    meta = {k: ck.get(k) for k in ("family", "task", "run", "seed", "width", "data_frac", "step", "size", "arch")}
    return ad, meta


class DMCBatch:
    """E episodes in lockstep through the evaluator's EnvPool."""

    def __init__(self, ad, task, seeds, workers, fake=False):
        self.ad, self.E = ad, len(seeds)
        S = EV.Settings(fake_render=fake)
        self.pool = EV.EnvPool(task, seeds, S, workers)
        self.Z = ad.encode(self.pool.frames0)
        self.ret = np.zeros(self.E)
        self.t_env = 0.0

    def latent(self):
        return self.Z

    def step(self, act):
        a = act.float().cpu().numpy()
        t0 = time.perf_counter()
        frames, r, done = self.pool.step(a)
        self.ret += r
        self.t_env += time.perf_counter() - t0
        self.Z = self.ad.filter_step(self.Z, a, frames)

    def close(self):
        self.pool.close()


def make_cost_factory(ad, P, dev, amp):
    def make_cost(Z):
        Zr = torch.as_tensor(np.asarray(Z), dtype=torch.float32, device=dev).repeat_interleave(P, 0)

        def cost(seq):
            with torch.autocast("cuda", dtype=torch.float16) if amp else torch.autocast("cpu", enabled=False):
                return ad.get_cost(Zr, seq.float())
        return cost
    return make_cost


def run_arm(ad, meta, task, arm, cfg, episodes, workers, dev, amp, T, fake, planner_seed=777, seed0=SEED0):
    A = EV.task_data(task).act_dim
    seeds = list(range(seed0, seed0 + episodes))
    env = DMCBatch(ad, task, seeds, workers, fake)
    gen = torch.Generator(device=dev).manual_seed(planner_seed)
    planner = PrefixMPPI(cfg, gen, dev)
    timer = Timer(dev)
    t0 = time.time()
    extra = {}
    m = re.fullmatch(r"hcpp(\d+)k(\d+)", arm)
    mp = re.fullmatch(r"pcpp(\d+)k(\d+)", arm)
    mr = re.fullmatch(r"rcpp(\d+)k(\d+)r(\d+)", arm)
    pe = parse_de(arm)
    if mr:                                      # hedge-then-refine (latency/refine.py, Section 17A)
        import math
        from latency.refine import run_reanchor
        d, K, iref = int(mr.group(1)), int(mr.group(2)), int(mr.group(3))
        q = max(1, math.ceil(d * iref / cfg.iterations))
        bank_d, bank_q = residual_bank(ad, task, d), residual_bank(ad, task, q)
        mc_d = make_cost_hedged(ad, cfg.population, dev, amp, d, bank_d, K, H)
        mc_q = make_cost_hedged(ad, cfg.population, dev, amp, q, bank_q, K, H, gen_seed=6)
        try:
            n_calls = run_reanchor(d, H, T, planner, mc_d, mc_q, mc_d, env, A, timer, iref)
        finally:
            env.close()
        pc = np.asarray(timer.per_call)
        return {"arm": arm, "returns": env.ret.tolist(), "return_mean": float(env.ret.mean()),
                "episodes": episodes, "seeds": [seeds[0], seeds[-1]], "decisions": T, "H": H,
                "n_calls": n_calls, "t_plan_s": float(pc.sum()), "t_env_s": env.t_env,
                "t_wall_s": time.time() - t0, "nonfinite_frac": planner.n_bad / max(planner.n_all, 1),
                "batch_size": episodes, "K": K, "q": q, "iters_ref": iref,
                "bank_rms_d": float(np.sqrt((bank_d ** 2).sum(1).mean())),
                "bank_rms_q": float(np.sqrt((bank_q ** 2).sum(1).mean()))}
    if mp:                                      # particle anchors, RSSM only (latency/ensemble.py)
        from latency.ensemble import make_cost_particles
        d, K = int(mp.group(1)), int(mp.group(2))
        make_cost = make_cost_particles(ad, cfg.population, dev, amp, d, K, H)
        sched = f"cpp{d}"
        extra = {"K": K, "anchors": "particles"}
    elif pe:                                      # delay-error model arms (latency/delay_error.py)
        kind, d, K, pen, lam = pe
        de = fit_delay_error(ad, meta, d, dev)
        make_cost = make_cost_delay_error(ad, cfg.population, dev, amp, de, kind, K, pen, lam, H)
        sched = f"cpp{d}"
        extra = {"kind": kind, "K": K, "penalty": pen, "lam": lam}
    elif m:                                       # residual-hedged committed prefix (latency/hedge.py)
        d, K = int(m.group(1)), int(m.group(2))
        bank = residual_bank(ad, task, d)
        make_cost = make_cost_hedged(ad, cfg.population, dev, amp, d, bank, K, H)
        sched = f"cpp{d}"
        extra = {"K": K, "bank_n": len(bank), "bank_rms": float(np.sqrt((bank ** 2).sum(1).mean()))}
    else:
        make_cost, sched = make_cost_factory(ad, cfg.population, dev, amp), arm
    try:
        n_calls = run_schedule(sched, H, T, planner, make_cost, env, A, timer)
    finally:
        env.close()
    pc = np.asarray(timer.per_call)
    return {"arm": arm, "returns": env.ret.tolist(), "return_mean": float(env.ret.mean()),
            "episodes": episodes, "seeds": [seeds[0], seeds[-1]], "decisions": T, "H": H,
            "n_calls": n_calls, "t_plan_s": float(pc.sum()), "t_plan_call_mean_s": float(pc.mean()),
            "t_plan_call_p95_s": float(np.percentile(pc, 95)), "t_env_s": env.t_env,
            "t_wall_s": time.time() - t0, "nonfinite_frac": planner.n_bad / max(planner.n_all, 1),
            "batch_size": episodes, **extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="study1")
    ap.add_argument("--arms", help="comma-separated arms, overrides --preset")
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=SEED0, help="first initial-state seed (tuning uses a disjoint range)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("DMC_ENV_WORKERS", 0)))
    ap.add_argument("--fake-render", action="store_true", help="smoke test without EGL")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    if amp:
        EV.check_precision(autocast_dtype=torch.float16)          # plan amendment A3
    ad, meta = load_adapter(a.ckpt, dev)
    task, run = meta["task"], meta["run"]
    cfg = MPPIConfig(population=a.pop, iterations=a.iters)
    arms = a.arms.split(",") if a.arms else PRESETS[a.preset]
    od = pathlib.Path(a.out) / meta["family"] / task / run / f"pop{a.pop}_it{a.iters}"
    od.mkdir(parents=True, exist_ok=True)
    failed = []
    for arm in arms:
        f = od / f"{arm}.json"
        if f.exists() and not os.environ.get("FORCE"):
            print("skip (done)", f, flush=True); continue
        try:
            r = run_arm(ad, meta, task, arm, cfg, a.episodes, a.workers, dev, amp, a.T, a.fake_render, seed0=a.seed0)
        except Exception:
            traceback.print_exc(); failed.append(arm); continue
        r.update(meta=meta, ckpt=str(a.ckpt), config=dataclasses.asdict(cfg),
                 gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", git=git_hash(),
                 host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"),
                 written=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        tmp = f.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(r, indent=1)); tmp.replace(f)
        print(f"{meta['family']}/{task}/{run} pop{a.pop} it{a.iters} {arm}: return {r['return_mean']:.1f} "
              f"({r['n_calls']} calls, {r['t_plan_call_mean_s'] * 1e3:.0f} ms/call at batch {a.episodes}, "
              f"{r['t_wall_s']:.0f}s)", flush=True)
    if failed:
        sys.exit(f"failed arms: {failed}")


if __name__ == "__main__":
    main()
