"""Section 19 analysis (EXPERIMENT_PLAN_SIDE.md, written before the runs): real-time normalised
return(htr) - return(cpp) over the 21 models of 7 cells, equal cell weight, hierarchical bootstrap
(models within cell, then episodes paired by seed), 10,000 resamples, one-sided, alpha = 0.05.
Secondary: htr vs rch, rch vs cpp, decisions per call, underruns, refined share.
Writes results/latency/wallclock_v4.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency.analyse import norm  # noqa: E402

W = ROOT / "results" / "latency" / "wallclock_v4"
CELLS = {c: [f"{c.replace('/', '_')}_seed{s}" for s in range(3)] for c in
         ("jepa/cup-catch", "tdmpc2/cup-catch", "rssm/walker-walk", "rssm/cheetah-run", "jepa/finger-spin",
          "tdmpc2/finger-spin", "tdmpc2/reacher-easy")}
B = 10_000


def load(run, arm):
    f = W / run / "pop512_it6" / f"{arm}.json"
    if not f.exists():
        return None, None
    r = json.load(open(f))
    eps = sorted(r["episodes"], key=lambda e: e["seed"])
    return norm(run.split("_")[1], [e["return"] for e in eps]), r


def contrast(a, b, rng):
    cells = []
    for c, runs in CELLS.items():
        ms = [load(r, a)[0] - load(r, b)[0] for r in runs if load(r, a)[0] is not None and load(r, b)[0] is not None]
        if ms:
            cells.append(np.stack(ms))
    est = float(np.mean([x.mean() for x in cells]))
    bs = np.zeros(B)
    for x in cells:
        m, n = x.shape
        mi = rng.integers(0, m, (B, m)); ei = rng.integers(0, n, (B, m, n))
        bs += x[mi[..., None], ei].mean((1, 2))
    bs /= len(cells)
    return {"est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_one_sided": float((bs <= 0).mean()), "models": int(sum(len(x) for x in cells)), "cells": len(cells)}


def main():
    rng = np.random.default_rng(20261022)
    out = {"confirmatory_htr_vs_cpp": contrast("htr", "cpp", rng),
           "secondary": {"htr_vs_rch": contrast("htr", "rch", rng), "rch_vs_cpp": contrast("rch", "cpp", rng)},
           "runs": {}}
    for c, runs in CELLS.items():
        for run in runs:
            out["runs"][run] = {}
            for arm in ("htr", "cpp", "rch"):
                x, r = load(run, arm)
                if r is None:
                    continue
                e = r["episodes"]
                out["runs"][run][arm] = {
                    "return_norm": float(x.mean()), "d_est": r["d_est"],
                    "call_decisions_median": float(np.median([q["call_decisions_median"] for q in e])),
                    "call_decisions_max": int(max(q["call_decisions_max"] for q in e)),
                    "underruns": int(sum(q["underruns"] for q in e)),
                    "refined_share": (float(sum(q.get("refined_calls", 0) for q in e) / max(sum(q["n_calls"] for q in e), 1))
                                      if arm == "htr" else None)}
    (ROOT / "results" / "latency" / "wallclock_v4.json").write_text(json.dumps(out, indent=1))
    v = out["confirmatory_htr_vs_cpp"]
    print(f"htr vs cpp (real time): {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] one-sided p = {v['p_one_sided']:.4f} ({v['models']} models)")
    for k, v in out["secondary"].items():
        print(f"  {k}: {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  p = {v['p_one_sided']:.4f}")
    for run, arms in out["runs"].items():
        print(f"  {run:28s}", {a: (round(v["return_norm"], 3), v["d_est"], v["call_decisions_max"], v["underruns"]) for a, v in arms.items()})


if __name__ == "__main__":
    main()
