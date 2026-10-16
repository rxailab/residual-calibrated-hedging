"""Study 15C analysis (paper/docs/EXPERIMENT_PLAN_SIDE.md Section 15C, written before the runs):
real-time return(rch) - return(cpp) over the six cup-catch models of JEPA and TD-MPC2, normalised
return, pooled with equal cell weight, hierarchical bootstrap (models within cell, then episodes
paired by initial-state seed), 10,000 resamples, one-sided, alpha = 0.05.  Secondary: cpp vs
aligned, realised decisions per call, underruns.  Writes results/latency/wallclock_v2.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency.analyse import norm  # noqa: E402

W = ROOT / "results" / "latency" / "wallclock_v2"
CELLS = {"jepa/cup-catch": [f"jepa_cup-catch_seed{s}" for s in range(3)],
         "tdmpc2/cup-catch": [f"tdmpc2_cup-catch_seed{s}" for s in range(3)]}
B = 10_000


def load(run, arm):
    f = W / run / "pop512_it6" / f"{arm}.json"
    if not f.exists():
        return None, None
    r = json.load(open(f))
    eps = sorted(r["episodes"], key=lambda e: e["seed"])
    return norm("cup-catch", [e["return"] for e in eps]), r


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
            "p_one_sided": float((bs <= 0).mean()), "models": int(sum(len(x) for x in cells))}


def main():
    rng = np.random.default_rng(20261016)
    out = {"confirmatory_rch_vs_cpp": contrast("rch", "cpp", rng),
           "secondary": {"cpp_vs_aligned": contrast("cpp", "aligned", rng),
                         "rch_vs_aligned": contrast("rch", "aligned", rng)}, "runs": {}}
    for c, runs in CELLS.items():
        for run in runs:
            out["runs"][run] = {}
            for arm in ("rch", "cpp", "aligned"):
                x, r = load(run, arm)
                if r is None:
                    continue
                out["runs"][run][arm] = {
                    "return_norm": float(x.mean()), "d_est": r["d_est"], "calibration": r.get("calibration"),
                    "startup_L95_s": r["startup_L95_s"],
                    "call_decisions_median": float(np.median([e["call_decisions_median"] for e in r["episodes"]])),
                    "call_decisions_max": int(max(e["call_decisions_max"] for e in r["episodes"])),
                    "underruns": int(sum(e["underruns"] for e in r["episodes"]))}
    (ROOT / "results" / "latency" / "wallclock_v2.json").write_text(json.dumps(out, indent=1))
    v = out["confirmatory_rch_vs_cpp"]
    print(f"rch vs cpp (real time): {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] one-sided p = {v['p_one_sided']:.4f}")
    for k, v in out["secondary"].items():
        print(f"  {k}: {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    for run, arms in out["runs"].items():
        print(run, {a: (round(v["return_norm"], 3), v["d_est"], v["call_decisions_max"], v["underruns"]) for a, v in arms.items()})


if __name__ == "__main__":
    main()
