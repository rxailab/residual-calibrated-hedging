"""Study 2 timing: single-environment planning-call latency (batch size 1).

For one model and every planner configuration (population x iterations) it
times the call actually made by each arm: paused (candidates of H decisions)
and cpp at d in {1, 2, 4, 8, 16} (prefix d plus H free decisions).  Each
call starts from a latent encoded from a D_eval frame and ends with the
committed actions on the host; CUDA is synchronised at both ends.  20 warm-up
calls, then --calls timed calls cycling over 20 states.

Writes results/latency/timing/<gpu>/<family>_<task>_<run>.json with L50, L95,
mean and every sample, plus GPU name, clocks and other processes on the GPU.
"""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.dmc import evaluate as EV  # noqa: E402
from latency.run_dmc import H, load_adapter, make_cost_factory, git_hash  # noqa: E402
from latency.schedule import MPPIConfig, PrefixMPPI  # noqa: E402
from paths import RESULTS  # noqa: E402

POPS, ITERS, DS = (64, 128, 256, 512), (2, 6), (0, 1, 2, 4, 8, 16)


def smi(q):
    try:
        return subprocess.run(["nvidia-smi"] + q, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def gpu_tag(name):
    n = name.lower()
    return "v100" if "v100" in n else "h200" if "h200" in n else "l40s" if "l40s" in n else \
        "l40" if "l40" in n else n.replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    dev = "cuda"
    EV.check_precision(autocast_dtype=torch.float16)
    ad, meta = load_adapter(a.ckpt, dev)
    td = EV.task_data(meta["task"])
    A = td.act_dim
    frames = np.stack([td.frames[EV.N_TRAIN + i, 50 + 10 * i] for i in range(20)])
    Zs = [ad.encode(frames[i:i + 1]) for i in range(20)]
    name = torch.cuda.get_device_name(0)
    out = {"meta": meta, "ckpt": a.ckpt, "gpu": name, "H": H, "calls": a.calls, "warmup": a.warmup,
           "clocks": smi(["-q", "-d", "CLOCK"]), "processes": smi(["--query-compute-apps=pid,used_memory",
                                                                  "--format=csv"]),
           "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"), "git": git_hash(),
           "written": None, "results": []}
    for P in POPS:
        for I in ITERS:
            cfg = MPPIConfig(population=P, iterations=I)
            planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(0), dev)
            make_cost = make_cost_factory(ad, P, dev, True)
            for d in DS:
                warm = torch.zeros((1, H, A), dtype=torch.float64, device=dev)
                prefix = torch.zeros((1, d, A), dtype=torch.float64, device=dev) if d else None
                ts = []
                for k in range(a.warmup + a.calls):
                    Z = Zs[k % 20]
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    mu = planner(make_cost(Z), warm, prefix)
                    act = mu[:, :max(d, 1)].float().cpu().numpy()  # noqa: F841  committed actions on host
                    torch.cuda.synchronize(); dt = time.perf_counter() - t0
                    if k >= a.warmup:
                        ts.append(dt)
                ts = np.asarray(ts)
                out["results"].append({"pop": P, "iters": I, "d": d, "L50_s": float(np.median(ts)),
                                       "L95_s": float(np.percentile(ts, 95)), "mean_s": float(ts.mean()),
                                       "samples_s": ts.tolist()})
                print(f"{meta['family']}/{meta['task']} pop{P} it{I} d{d}: L50 {np.median(ts) * 1e3:.1f} ms, "
                      f"L95 {np.percentile(ts, 95) * 1e3:.1f} ms", flush=True)
    out["written"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    od = RESULTS / "latency" / "timing" / gpu_tag(name)
    od.mkdir(parents=True, exist_ok=True)
    f = od / f"{meta['family']}_{meta['task']}_{meta['run']}.json"
    tmp = f.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(out, indent=1)); tmp.replace(f)
    print("wrote", f)


if __name__ == "__main__":
    main()
