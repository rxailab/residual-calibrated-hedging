"""Batch-1 latency of LeWM planning calls (Study 15B, secondary): LeWM's CEM (300 samples, 30
iterations), horizon 5 blocks, for the paused call (5 blocks), committed prefix (d + 5) and RCH
(K = 4 anchors, batched) at d in {2, 4}; 5 warm-up and 30 timed calls, CUDA-synchronised, from
encoded dataset frames.  Converts to decisions with each task's decision period (5 environment
steps: PushT 0.1 s, Cube 0.05 s, Reacher 0.02 s per step).
Writes results/latency/lewm/timing_<gpu>.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import lewm as L  # noqa: E402

STEP_S = {"pusht": 0.1, "cube": 0.05, "reacher": 0.02}


def main():
    dev = "cuda"
    out = {"gpu": torch.cuda.get_device_name(0), "host": os.uname().nodename, "results": []}
    for task in ("pusht", "cube", "reacher"):
        model = L.load_model(task, dev)
        ds = L.load_dataset(task)
        tf = L.img_transform()
        ch = ds.load_chunk(np.array([int(ds.get_col_data("episode_idx" if "episode_idx" in ds.column_names else "ep_idx")[0])]),
                           np.array([0]), np.array([30]))
        z0 = L.encode(model, tf(ch[0]["pixels"][0])[None].to(dev))
        g = L.encode(model, tf(ch[0]["pixels"][-1])[None].to(dev))
        D = z0.shape[1]
        A = model.action_encoder.patch_embed.in_channels          # action block dimension (5 steps)
        gen = torch.Generator(device=dev).manual_seed(0)
        configs = [("paused", 0, None)]
        for d in (2, 4):
            configs += [("cpp", d, None), ("rch", d, 4)]
        for name, d, K in configs:
            pre = torch.zeros((1, d, A), device=dev) if d else None
            anc = torch.zeros((1, K, D), device=dev) if K else None
            ts = []
            for i in range(35):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                L.cem(L.cost_fn(model, z0, g, pre, anc), torch.zeros((1, L.H, A), device=dev), gen, dev)
                torch.cuda.synchronize()
                if i >= 5:
                    ts.append(time.perf_counter() - t0)
            ts = np.asarray(ts)
            dec = 5 * STEP_S[task]
            r = {"task": task, "call": name, "d": d, "K": K, "L50_s": float(np.median(ts)), "L95_s": float(np.percentile(ts, 95)),
                 "decision_s": dec, "delay_decisions": int(np.ceil(np.percentile(ts, 95) / dec))}
            out["results"].append(r)
            print(f"{task} {name} d{d} K{K}: L50 {r['L50_s'] * 1e3:.0f} ms, L95 {r['L95_s'] * 1e3:.0f} ms -> {r['delay_decisions']} decisions", flush=True)
    tag = out["gpu"].replace("NVIDIA ", "").replace(" ", "_").lower()
    (L.OUT / f"timing_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
