"""Pack one DMControl task's shards into training-ready arrays.

frames.npy   (2000, 501, 64, 64, 3) uint8 memmap: the stored 128x128 frames
             averaged over exact 2x2 blocks (deterministic, no resampling filter)
meta.npz     actions, rewards, proprio, qpos, qvel, MuJoCo integration states,
             and per-episode eps tier, seed and return

Episode e sits at index e (shard s holds episodes 50s .. 50s+49); episodes
1900-1999 are D_eval.  Everything is regenerable from the shards, so it lives
on scratch.
"""
import sys, pathlib, time, argparse
import numpy as np

import os
_SCR = pathlib.Path(os.environ.get("RCH_SCRATCH", "/scratch/hpc/11/xiar3/MetricTrust"))   # side paper 16B
SRC = _SCR / "data" / "dmc"
DST = _SCR / "data" / "dmc64"


def pack(task, n_ep=2000, per=50):
    shards = sorted((SRC / task).glob("shard_*.npz"))
    assert len(shards) == n_ep // per, f"{task}: {len(shards)} shards"
    out = DST / task; out.mkdir(parents=True, exist_ok=True)
    frames = None; meta = {}
    t0 = time.time()
    for s, f in enumerate(shards):
        z = np.load(f)
        assert int(z["episode"][0]) == s * per, (f, z["episode"][:3])
        px = z["pixels"]                                   # (50, 501, 128, 128, 3)
        small = px.reshape(*px.shape[:2], 64, 2, 64, 2, 3).mean(axis=(3, 5))
        small = np.rint(small).astype(np.uint8)
        if frames is None:
            frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8,
                                               shape=(n_ep, *small.shape[1:]))
        frames[s * per:(s + 1) * per] = small
        for k in ("action", "reward", "proprio", "qpos", "qvel", "mjstate", "eps", "seed", "ret", "episode"):
            meta.setdefault(k, []).append(z[k])
        print(f"  {task} shard {s:2d} packed ({time.time()-t0:.0f}s)", flush=True)
    frames.flush()
    np.savez(out / "meta.npz", **{k: np.concatenate(v) for k, v in meta.items()})
    m = np.load(out / "meta.npz")
    assert (m["episode"] == np.arange(n_ep)).all()
    print(f"{task}: frames {frames.shape}, meta keys {sorted(m.files)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True)
    pack(ap.parse_args().task)
