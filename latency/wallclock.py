"""Study 4: wall-clock asynchronous control, to test the delay-in-decisions
abstraction.

A simulator process steps one DMControl episode on a real-time clock (one
decision every dt seconds of wall time), executing whatever action the shared
buffer holds for the current decision (the last executed action if none has
been written, counted as an underrun).  The planner (this process, GPU) loops:
filter the latent through every frame received so far, plan, write the plan
into the buffer.  With d_est = ceil(L95 / dt) from a start-up latency
measurement:
  cpp      the candidates' first d_est actions are the buffer's actions for
           decisions n .. n + d_est - 1 (n = decisions executed when the call
           starts); the plan is written from decision n + d_est on.
  aligned  plan from the latent at n with no prefix (d_est + H long); positions
           d_est onward are written from decision n + d_est on.
  rch      cpp with residual-calibrated hedging (latency/hedge.py, K = 4 anchors
           scored in one batched call; plan Section 15C).
  htr      hedge-then-refine (latency/refine.py; plan Section 19): rch's first I - 1
           iterations from the latent at the call's start; then the frames that arrived
           meanwhile are filtered and the last iteration is re-anchored on the latest
           latent, with the committed actions still to run before the plan takes effect
           as its prefix (skipped if none remain).
With --calibrate (Section 15C), d_est is then re-set from one unscored real-time
episode of the slowest arm: the 95th percentile of the decisions that elapsed
during each call, rounded up, and results go to results/latency/wallclock_v2.
Plan actions for decisions that have already executed when the call returns
are dropped.  The pre-episode plan is computed before the clock starts.

Writes results/latency/wallclock/<family>_<task>_<run>/pop<P>_it<I>/<arm>.json
with returns, realised call durations (in seconds and decisions), underruns
and clock overruns.
"""
import argparse
import datetime
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.dmc import evaluate as EV  # noqa: E402
from latency.run_dmc import H, SEED0, load_adapter, make_cost_factory, git_hash  # noqa: E402
from latency.schedule import MPPIConfig, PrefixMPPI, shift  # noqa: E402
from latency.hedge import make_cost_hedged, residual_bank  # noqa: E402
from latency.refine import mppi_run  # noqa: E402
from paths import RESULTS  # noqa: E402

DT = {"cartpole-swingup": 0.02, "walker-walk": 0.05, "cup-catch": 0.04}   # action repeat 2


def sim_proc(task, seed, T, dt, act_buf, written, executed, start_evt, conn, fake):
    S = EV.Settings(fake_render=fake)
    env = EV.DMCEnv(task, img=128, render=not fake)
    conn.send(EV.frame(env.reset(seed), S))
    A = len(act_buf) // T
    start_evt.wait()
    t_next = time.monotonic()
    last = np.zeros(A)
    underruns, overruns = 0, 0
    for n in range(T):
        now = time.monotonic()
        if now < t_next:
            time.sleep(t_next - now)
        elif now - t_next > dt:
            overruns += 1
        if written[n]:
            a = np.frombuffer(act_buf.get_obj())[n * A:(n + 1) * A].copy()
        else:
            a, underruns = last, underruns + 1
        o, r, _ = env.step(a)
        last = a
        executed.value = n + 1
        conn.send((n, EV.frame(o, S), float(r), a))
        t_next += dt
    conn.send(("end", underruns, overruns))


def episode(ad, task, seed, arm, cfg, d_est, T, dev, fake, bank=None):
    A = EV.task_data(task).act_dim
    dt = DT[task]
    ctx = mp.get_context("spawn")
    act_buf = ctx.Array("d", T * A)
    written = ctx.Array("b", T, lock=False)
    executed = ctx.Value("i", 0)
    start_evt = ctx.Event()
    a_conn, b_conn = ctx.Pipe()
    p = ctx.Process(target=sim_proc, args=(task, seed, T, dt, act_buf, written, executed, start_evt, b_conn, fake),
                    daemon=True)
    p.start()
    frame0 = a_conn.recv()
    Z = ad.encode(frame0[None])
    planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(777), dev)
    if arm == "rch":
        make_cost = make_cost_hedged(ad, cfg.population, dev, dev == "cuda", d_est, bank, 4, H, batch_anchors=True)
    elif arm == "htr":                           # bank: {q: q-step residual bank} for q = 1 .. d_est
        make_cost = make_cost_hedged(ad, cfg.population, dev, dev == "cuda", d_est, bank[d_est], 4, H,
                                     batch_anchors=True)
        make_cost_q = {q: make_cost_hedged(ad, cfg.population, dev, dev == "cuda", q, bank[q], 4, H,
                                           batch_anchors=True, gen_seed=6) for q in range(1, d_est + 1)}
    else:
        make_cost = make_cost_factory(ad, cfg.population, dev, dev == "cuda")
    buf = np.frombuffer(act_buf.get_obj())

    def write(plan, start):
        for j in range(plan.shape[0]):
            n = start + j
            if n >= T or n < executed.value:
                continue
            with act_buf.get_lock():
                buf[n * A:(n + 1) * A] = plan[j]
            written[n] = 1

    mu0 = planner(make_cost(Z), torch.zeros((1, H, A), dtype=torch.float64, device=dev))
    write(mu0[0].float().cpu().numpy(), 0)
    w = shift(mu0, d_est) if arm in ("cpp", "rch", "htr") else torch.cat([mu0, torch.zeros_like(mu0[:, :d_est])], 1)
    st = {"ret": 0.0, "seen": 0, "Z": Z, "end": None}
    calls, refined = [], 0
    start_evt.set()

    def poll():                                  # filter every frame received so far
        while a_conn.poll():
            msg = a_conn.recv()
            if msg[0] == "end":
                st["end"] = msg; return
            n, fr, r, a = msg
            st["ret"] += r; st["seen"] = n + 1
            st["Z"] = ad.filter_step(st["Z"], a[None], fr[None])

    while st["end"] is None:
        poll()
        if st["end"] is not None or st["seen"] >= T:
            continue
        n0, Z = st["seen"], st["Z"]
        t0 = time.monotonic()
        if arm == "htr":
            pre = buf.reshape(T, A)[n0:n0 + d_est].copy()
            if len(pre) < d_est:
                pre = np.concatenate([pre, np.repeat(pre[-1:] if len(pre) else np.zeros((1, A)), d_est - len(pre), 0)])
            mu, sd = mppi_run(planner, make_cost(Z), w, torch.full_like(w, cfg.init_std), cfg.iterations - 1,
                              torch.as_tensor(pre[None], dtype=torch.float64, device=dev))
            poll()                               # frames that arrived during the first iterations
            q = n0 + d_est - st["seen"]
            if st["end"] is None and 1 <= q <= d_est and n0 + d_est <= T:
                pre2 = buf.reshape(T, A)[st["seen"]:n0 + d_est].copy()
                mu, sd = mppi_run(planner, make_cost_q[q](st["Z"]), mu, sd, 1,
                                  torch.as_tensor(pre2[None], dtype=torch.float64, device=dev))
                refined += 1
            plan = mu[0].float().cpu().numpy()
        elif arm in ("cpp", "rch"):
            pre = buf.reshape(T, A)[n0:n0 + d_est].copy()
            if len(pre) < d_est:
                pre = np.concatenate([pre, np.repeat(pre[-1:] if len(pre) else np.zeros((1, A)), d_est - len(pre), 0)])
            mu = planner(make_cost(Z), w, torch.as_tensor(pre[None], dtype=torch.float64, device=dev))
            plan = mu[0].float().cpu().numpy()
        else:
            mu = planner(make_cost(Z), w)
            plan = mu[0, d_est:].float().cpu().numpy()
        el = time.monotonic() - t0
        write(plan, n0 + d_est)
        calls.append({"start": n0, "end": int(executed.value), "seconds": el})
        w = shift(mu, d_est)
    p.join(timeout=30)
    ret, end = st["ret"], st["end"]
    lat = np.array([c["seconds"] for c in calls]) if calls else np.zeros(1)
    late = np.array([c["end"] - c["start"] for c in calls]) if calls else np.zeros(1)
    return {"seed": seed, "return": ret, "underruns": end[1], "clock_overruns": end[2], "n_calls": len(calls),
            "call_s_median": float(np.median(lat)), "call_s_p95": float(np.percentile(lat, 95)),
            "call_decisions_median": float(np.median(late)), "call_decisions_p95": float(np.percentile(late, 95)),
            "call_decisions_max": int(late.max()), "refined_calls": refined}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--arms", default="cpp,aligned")
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--fake-render", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="re-set d_est from one unscored real-time episode")
    ap.add_argument("--seed0", type=int, default=SEED0, help="first initial-state seed (16A: 950,100)")
    ap.add_argument("--out-sub", default=None, help="results/latency/<out-sub> (default wallclock or wallclock_v2)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        EV.check_precision(autocast_dtype=torch.float16)
    ad, meta = load_adapter(a.ckpt, dev)
    task = meta["task"]
    A = EV.task_data(task).act_dim
    cfg = MPPIConfig(population=a.pop, iterations=a.iters)
    # start-up latency of the cpp call at batch 1 (prefix of an initial guess of d, then refit once)
    planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(0), dev)
    make_cost = make_cost_factory(ad, a.pop, dev, dev == "cuda")
    td = EV.task_data(task)
    Z = ad.encode(np.stack([td.frames[EV.N_TRAIN, 60]]))
    d_est = 1
    for _ in range(2):
        ts = []
        for k in range(40):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            mu = planner(make_cost(Z), torch.zeros((1, H, A), dtype=torch.float64, device=dev),
                         torch.zeros((1, d_est, A), dtype=torch.float64, device=dev))
            mu.cpu()
            if k >= 10:
                ts.append(time.perf_counter() - t0)
        L50, L95 = float(np.median(ts)), float(np.percentile(ts, 95))
        d_est = max(1, math.ceil(L95 / DT[task]))
    print(f"start-up latency L50 {L50 * 1e3:.1f} ms, L95 {L95 * 1e3:.1f} ms -> d_est {d_est}", flush=True)
    arms = a.arms.split(",")
    calib = None
    if a.calibrate:
        slow = "htr" if "htr" in arms else ("rch" if "rch" in arms else arms[0])
        bank0 = {q: residual_bank(ad, task, q) for q in range(1, d_est + 1)} if slow == "htr" else \
            (residual_bank(ad, task, d_est) if slow == "rch" else None)
        ep = episode(ad, task, SEED0 + 10_000, slow, cfg, d_est, a.T, dev, a.fake_render, bank0)
        d_cal = max(1, math.ceil(ep["call_decisions_p95"]))
        calib = {"arm": slow, "d_startup": d_est, "call_decisions_p95": ep["call_decisions_p95"],
                 "call_decisions_max": ep["call_decisions_max"], "d_est": d_cal}
        print(f"calibration episode ({slow}): p95 {ep['call_decisions_p95']:.1f} decisions per call -> d_est {d_cal}",
              flush=True)
        d_est = d_cal
    bank = residual_bank(ad, task, d_est) if "rch" in arms else None
    banks_htr = {q: residual_bank(ad, task, q) for q in range(1, d_est + 1)} if "htr" in arms else None
    sub = a.out_sub or ("wallclock_v2" if a.calibrate else "wallclock")
    od = RESULTS / "latency" / sub / f"{meta['family']}_{task}_{meta['run']}" / f"pop{a.pop}_it{a.iters}"
    od.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        f = od / f"{arm}.json"
        if f.exists() and not os.environ.get("FORCE"):
            print("skip (done)", f); continue
        eps = []
        for i in range(a.episodes):
            eps.append(episode(ad, task, a.seed0 + i, arm, cfg, d_est, a.T, dev, a.fake_render,
                               banks_htr if arm == "htr" else bank))
            print(f"{arm} seed {a.seed0 + i}: return {eps[-1]['return']:.1f}, calls {eps[-1]['n_calls']}, "
                  f"median {eps[-1]['call_decisions_median']:.1f} decisions, underruns {eps[-1]['underruns']}",
                  flush=True)
        r = {"arm": arm, "d_est": d_est, "calibration": calib, "startup_L50_s": L50, "startup_L95_s": L95, "dt": DT[task],
             "episodes": eps, "return_mean": float(np.mean([e["return"] for e in eps])), "meta": meta,
             "ckpt": a.ckpt, "pop": a.pop, "iters": a.iters, "H": H,
             "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "git": git_hash(),
             "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"),
             "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        f.write_text(json.dumps(r, indent=1)); print("wrote", f, flush=True)


if __name__ == "__main__":
    main()
