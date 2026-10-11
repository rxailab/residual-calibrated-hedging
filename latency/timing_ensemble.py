"""Batch-1 latency, memory and parameters of the ensemble baseline against RCH
(plan Section 14).  One family-task cell: its three member models.

Times one committed-prefix call (prefix d, H free decisions) at batch size 1,
10 warm-up and 50 timed calls, CUDA-synchronised, for
  single model:  K = 1 (committed prefix) and K = 4 (RCH, batched anchors);
  ensemble:      M = 3 members evaluated one after another, at 6 iterations
                 (same population and iterations) and at 2 iterations;
                 ensemble + RCH (K = 4 per member, batched anchors), 6 iterations;
  RSSM only:     K = 4 particle anchors (batched).
Also records each configuration's peak GPU memory above the loaded weights,
each member's parameter count and weight memory, and the per-decision cost of
filtering the latent (one member against all three) at batch size 1.
Writes results/latency/timing_ensemble/<gpu>/<family>_<task>.json.
"""
import argparse, json, os, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, torch  # noqa: E402
from experiments.dmc import evaluate as EV  # noqa: E402
from latency.run_dmc import H, load_adapter  # noqa: E402
from latency.hedge import make_cost_hedged, residual_bank  # noqa: E402
from latency.ensemble import make_cost_ensemble, make_cost_particles, member_ckpts  # noqa: E402
from latency.schedule import MPPIConfig, PrefixMPPI  # noqa: E402
from latency.timing import gpu_tag  # noqa: E402
from paths import RESULTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("family"); ap.add_argument("task")
ap.add_argument("--ckpt-root", default=os.environ.get("CKPT_KEEP",
                                                      "/mmfs1/storage/users/xiar3/exp/MetricTrust/checkpoints_keep"))
ap.add_argument("--calls", type=int, default=50)
ap.add_argument("--pop", type=int, default=512)
a = ap.parse_args()
dev = "cuda"
EV.check_precision(autocast_dtype=torch.float16)
td = EV.task_data(a.task); A = td.act_dim
frame = np.stack([td.frames[EV.N_TRAIN, 60]])
torch.cuda.reset_peak_memory_stats()
base0 = torch.cuda.memory_allocated()
ads, metas, weights = [], [], []
for c in member_ckpts(a.ckpt_root, a.family, a.task):
    m0 = torch.cuda.memory_allocated()
    ad, meta = load_adapter(c, dev)
    ads.append(ad); metas.append(meta)
    weights.append({"run": meta["run"], "params": int(sum(p.numel() for p in ad.m.parameters())),
                    "weight_bytes_gpu": int(torch.cuda.memory_allocated() - m0)})
Zs = [ad.encode(frame) for ad in ads]
out = {"family": a.family, "task": a.task, "gpu": torch.cuda.get_device_name(0), "host": os.uname().nodename,
       "job": os.environ.get("SLURM_JOB_ID"), "members": weights, "results": []}


def timed(make_cost, Z, cfg, d):
    planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(0), dev)
    warm = torch.zeros((1, H, A), dtype=torch.float64, device=dev)
    pre = torch.zeros((1, d, A), dtype=torch.float64, device=dev)
    ts = []
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    for k in range(10 + a.calls):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        mu = planner(make_cost(Z), warm, pre); mu[:, :d].float().cpu().numpy()
        torch.cuda.synchronize()
        if k >= 10:
            ts.append(time.perf_counter() - t0)
    return np.asarray(ts), int(torch.cuda.max_memory_allocated() - base)


for d in (4, 8):
    banks = [residual_bank(ad, a.task, d) for ad in ads]
    configs = [("single_K1", 6, lambda: make_cost_hedged(ads[0], a.pop, dev, True, d, banks[0], 1, H, batch_anchors=True), Zs[0]),
               ("single_K4", 6, lambda: make_cost_hedged(ads[0], a.pop, dev, True, d, banks[0], 4, H, batch_anchors=True), Zs[0]),
               ("ensemble_M3", 6, lambda: make_cost_ensemble(ads, a.pop, dev, True, d, H), Zs),
               ("ensemble_M3", 2, lambda: make_cost_ensemble(ads, a.pop, dev, True, d, H), Zs),
               ("ensemble_M3_K4", 6, lambda: make_cost_ensemble(ads, a.pop, dev, True, d, H, banks, 4, batch_anchors=True), Zs)]
    if a.family == "rssm":
        configs.append(("particles_K4", 6, lambda: make_cost_particles(ads[0], a.pop, dev, True, d, 4, H, batch_anchors=True), Zs[0]))
    for name, iters, build, Z in configs:
        ts, peak = timed(build(), Z, MPPIConfig(population=a.pop, iterations=iters), d)
        out["results"].append({"config": name, "d": d, "pop": a.pop, "iters": iters, "L50_s": float(np.median(ts)),
                               "L95_s": float(np.percentile(ts, 95)), "peak_extra_bytes": peak})
        print(f"{a.family}/{a.task} d{d} {name} it{iters}: L50 {np.median(ts) * 1e3:.1f} ms "
              f"L95 {np.percentile(ts, 95) * 1e3:.1f} ms, peak +{peak / 2 ** 20:.0f} MiB", flush=True)

# per-decision latent filtering at batch size 1: one member against all three
act = np.zeros((1, A))
fr = np.stack([td.frames[EV.N_TRAIN, 61]])
filt = {}
for name, members in (("single", ads[:1]), ("ensemble_M3", ads)):
    for ad in members:
        ad.encode(frame)                                  # resets TD-MPC2's frame history
    Zf = [ad.encode(frame) for ad in members]
    ts = []
    for k in range(10 + a.calls):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        Zf = [ad.filter_step(z, act, fr) for ad, z in zip(members, Zf)]
        torch.cuda.synchronize()
        if k >= 10:
            ts.append(time.perf_counter() - t0)
    filt[name] = {"L50_s": float(np.median(ts)), "L95_s": float(np.percentile(ts, 95))}
    print(f"{a.family}/{a.task} filter {name}: L50 {np.median(ts) * 1e3:.2f} ms", flush=True)
out["filter"] = filt
od = RESULTS / "latency" / "timing_ensemble" / gpu_tag(out["gpu"]); od.mkdir(parents=True, exist_ok=True)
(od / f"{a.family}_{a.task}.json").write_text(json.dumps(out, indent=1))
