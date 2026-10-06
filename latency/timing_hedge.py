"""Single-environment call latency of residual-hedged committed-prefix MPPI
(latency/hedge.py) against K = 1, for the method decision: does hedging over K
anchors cost latency at batch size 1, where the GPU has slack?

Times the cpp call (prefix d, H free decisions) for K in {1, 4, 16} and a few
planner configurations; 10 warm-up and 50 timed calls, CUDA-synchronised.
Writes results/latency/timing_hedge/<gpu>/<family>_<task>_<run>.json.
"""
import argparse, json, os, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, torch  # noqa: E402
from experiments.dmc import evaluate as EV  # noqa: E402
from latency.run_dmc import H, load_adapter  # noqa: E402
from latency.hedge import make_cost_hedged, residual_bank  # noqa: E402
from latency.schedule import MPPIConfig, PrefixMPPI  # noqa: E402
from latency.timing import gpu_tag  # noqa: E402
from paths import RESULTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--calls", type=int, default=50)
a = ap.parse_args()
dev = "cuda"
EV.check_precision(autocast_dtype=torch.float16)
ad, meta = load_adapter(a.ckpt, dev)
td = EV.task_data(meta["task"]); A = td.act_dim
Z = ad.encode(np.stack([td.frames[EV.N_TRAIN, 60]]))
out = {"meta": meta, "gpu": torch.cuda.get_device_name(0), "host": os.uname().nodename, "results": []}
for d in (2, 4, 8):
    bank = residual_bank(ad, meta["task"], d)
    for P, I in ((512, 6), (128, 6), (32, 6)):
        cfg = MPPIConfig(population=P, iterations=I)
        for K in (1, 4, 16):
            mc = make_cost_hedged(ad, P, dev, True, d, bank, K, H, batch_anchors=True)
            planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(0), dev)
            warm = torch.zeros((1, H, A), dtype=torch.float64, device=dev)
            pre = torch.zeros((1, d, A), dtype=torch.float64, device=dev)
            ts = []
            for k in range(10 + a.calls):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                mu = planner(mc(Z), warm, pre); mu[:, :d].float().cpu().numpy()
                torch.cuda.synchronize()
                if k >= 10:
                    ts.append(time.perf_counter() - t0)
            ts = np.asarray(ts)
            out["results"].append({"d": d, "pop": P, "iters": I, "K": K, "L50_s": float(np.median(ts)),
                                   "L95_s": float(np.percentile(ts, 95))})
            print(f"{meta['family']}/{meta['task']} d{d} pop{P} it{I} K{K}: L50 {np.median(ts) * 1e3:.1f} ms "
                  f"L95 {np.percentile(ts, 95) * 1e3:.1f} ms", flush=True)
od = RESULTS / "latency" / "timing_hedge" / gpu_tag(out["gpu"]); od.mkdir(parents=True, exist_ok=True)
(od / f"{meta['family']}_{meta['task']}_{meta['run']}.json").write_text(json.dumps(out, indent=1))
