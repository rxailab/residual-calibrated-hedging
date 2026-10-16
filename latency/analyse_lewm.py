"""Study 15B analysis (paper/docs/EXPERIMENT_PLAN_SIDE.md Section 15B and its amendment, written
before the confirmatory runs).

Confirmatory (amendment 2): success(rch_d k4) - success(cpp_d) in the hand-over setting H (period 5,
the first d decisions execute the recorded actions, goal offset 25 + 5d, budget 5d + 50), pooled over
pusht, cube and reacher with equal weight, at d = 2 and d = 4; one-sided, Holm over the two delays;
paired bootstrap over episodes within task (the same episode indices for both arms), 10,000
resamples; one-sided p = share of resampled pooled differences <= 0.
Secondary: the same in S1 (LeWM's protocol: goal offset 25, budget 50); cpp vs aligned; mismatch loss
paused5 - X_d; per-task success rates.
Writes results/latency/lewm_summary.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency.analyse_hedge_full import holm  # noqa: E402

D = ROOT / "results" / "latency" / "lewm"
TASKS = ("pusht", "cube", "reacher")
SETTINGS = {"H": {2: "r5_g35_b60_h2", 4: "r5_g45_b70_h4"}, "S1": {2: "r5_g25_b50", 4: "r5_g25_b50"}}
B = 10_000


def load(setting, task, arm):
    f = D / setting / task / f"{arm}.json"
    return np.asarray(json.load(open(f))["successes"], float) if f.exists() else None


def contrast(setting, a, b, rng, tasks=TASKS):
    diffs = [(t, load(setting, t, a), load(setting, t, b)) for t in tasks]
    diffs = [(t, x - y) for t, x, y in diffs if x is not None and y is not None]
    if not diffs:
        return None
    est = float(np.mean([x.mean() for _, x in diffs]))
    bs = np.zeros(B)
    for _, x in diffs:
        bs += x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    bs /= len(diffs)
    return {"est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_one_sided": float((bs <= 0).mean()), "tasks": [t for t, _ in diffs],
            "per_task": {t: float(x.mean()) for t, x in diffs}}


def main():
    rng = np.random.default_rng(20261016)
    out = {"levels": {}, "confirmatory": {}, "secondary": {}}
    for sname, ss in SETTINGS.items():
        out["levels"][sname] = {f"d{d}": {t: {a: (float(v.mean()) if (v := load(ss[d], t, a)) is not None else None)
                                             for a in ("paused5", f"naive{d}", f"aligned{d}", f"cpp{d}", f"rch{d}k4")}
                                          for t in TASKS} for d in (2, 4)}
    conf = {f"rch_vs_cpp_d{d}": contrast(SETTINGS["H"][d], f"rch{d}k4", f"cpp{d}", rng) for d in (2, 4)}
    if all(conf.values()):
        for k, v in zip(conf, holm([conf[k]["p_one_sided"] for k in conf])):
            conf[k]["p_holm"] = v
    out["confirmatory"] = conf
    sec = out["secondary"]
    for d in (2, 4):
        sec[f"S1_rch_vs_cpp_d{d}"] = contrast(SETTINGS["S1"][d], f"rch{d}k4", f"cpp{d}", rng)
        for sname, ss in SETTINGS.items():
            sec[f"{sname}_cpp_vs_aligned_d{d}"] = contrast(ss[d], f"cpp{d}", f"aligned{d}", rng)
            for arm in (f"cpp{d}", f"rch{d}k4", f"aligned{d}"):
                sec[f"{sname}_M_{arm}"] = contrast(ss[d], "paused5", arm, rng)
    (ROOT / "results" / "latency" / "lewm_summary.json").write_text(json.dumps(out, indent=1))
    for sname in SETTINGS:
        for dd, lvd in out["levels"][sname].items():
            print(sname, dd)
            for t, lv in lvd.items():
                print(f"  {t:8s}", "  ".join(f"{a} {v:.3f}" for a, v in lv.items() if v is not None))
    for sect in ("confirmatory", "secondary"):
        print(sect)
        for k, v in out[sect].items():
            if v:
                ph = f", Holm p = {v['p_holm']:.4f}" if "p_holm" in v else ""
                print(f"  {k:24s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  one-sided p = "
                      f"{v['p_one_sided']:.4f}{ph}  {v['per_task']}")


if __name__ == "__main__":
    main()
