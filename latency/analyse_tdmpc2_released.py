"""Study 17B analysis (EXPERIMENT_PLAN_SIDE.md Section 17B, written before the runs): released TD-MPC2
models under a pipelined delay.  Normalised return with the pixel study's reference pairs; a task is
eligible if its mean paused1 return is at least 0.15.  Confirmatory: R(rch_d k4) - R(cpp_d) at d = 4
and d = 8, pooled with equal task weight, hierarchical bootstrap (models within task, then episodes,
shared indices across arms), one-sided, Holm over the two delays.  Secondary: d = 2, cpp vs naive,
mismatch loss paused1 - X_d, per task.  cartpole-swingup seed 1 is excluded (pre-release layout).
Writes results/latency/tdmpc2_released.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency.analyse import norm, FLOOR  # noqa: E402
from latency.analyse_hedge_full import holm  # noqa: E402

D = ROOT / "results" / "latency" / "tdmpc2_released"
TASKS = ("cartpole-swingup", "walker-walk", "cup-catch", "reacher-easy", "cheetah-run", "finger-spin")
B = 10_000


def load(task, arm):
    xs = []
    for s in (1, 2, 3):
        f = D / task / f"seed{s}" / f"{arm}.json"
        if f.exists():
            xs.append(norm(task, json.load(open(f))["returns"]))
    return np.stack(xs) if xs else None


def contrast(tasks, a, b, rng):
    cells = []
    for t in tasks:
        x, y = load(t, a), load(t, b)
        if x is not None and y is not None and len(x) == len(y):
            cells.append(x - y)
    est = float(np.mean([c.mean() for c in cells]))
    bs = np.zeros(B)
    for x in cells:
        m, n = x.shape
        mi = rng.integers(0, m, (B, m)); ei = rng.integers(0, n, (B, m, n))
        bs += x[mi[..., None], ei].mean((1, 2))
    bs /= len(cells)
    return {"est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_one_sided": float((bs <= 0).mean()), "tasks": len(cells)}


def main():
    rng = np.random.default_rng(20261019)
    arms = ["paused1"] + [f"{k}{d}" for d in (2, 4, 8) for k in ("naive", "cpp")] + [f"rch{d}k4" for d in (2, 4, 8)]
    levels = {t: {a: (float(v.mean()) if (v := load(t, a)) is not None else None) for a in arms} for t in TASKS}
    eligible = [t for t in TASKS if levels[t]["paused1"] is not None and levels[t]["paused1"] >= FLOOR]
    out = {"levels": levels, "eligible": eligible, "confirmatory": {}, "secondary": {}}
    conf = {f"rch_vs_cpp_d{d}": contrast(eligible, f"rch{d}k4", f"cpp{d}", rng) for d in (4, 8)}
    for k, v in zip(conf, holm([conf[k]["p_one_sided"] for k in conf])):
        conf[k]["p_holm"] = v
    out["confirmatory"] = conf
    sec = out["secondary"]
    sec["rch_vs_cpp_d2"] = contrast(eligible, "rch2k4", "cpp2", rng)
    for d in (2, 4, 8):
        sec[f"cpp_vs_naive_d{d}"] = contrast(eligible, f"cpp{d}", f"naive{d}", rng)
        for arm in (f"naive{d}", f"cpp{d}", f"rch{d}k4"):
            sec[f"M_{arm}"] = contrast(eligible, "paused1", arm, rng)
    out["per_task_rch_minus_cpp"] = {t: {d: levels[t][f"rch{d}k4"] - levels[t][f"cpp{d}"] for d in (2, 4, 8)
                                         if levels[t][f"rch{d}k4"] is not None} for t in eligible}
    (ROOT / "results" / "latency" / "tdmpc2_released.json").write_text(json.dumps(out, indent=1))
    print("eligible:", eligible)
    for t in TASKS:
        print(f"  {t:17s}", "  ".join(f"{a} {v:.3f}" for a, v in levels[t].items() if v is not None))
    for sect in ("confirmatory", "secondary"):
        print(sect)
        for k, v in out[sect].items():
            ph = f", Holm p = {v['p_holm']:.4f}" if "p_holm" in v else ""
            print(f"  {k:18s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  one-sided p = {v['p_one_sided']:.4f}{ph}")


if __name__ == "__main__":
    main()
