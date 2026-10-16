"""Study 16C analysis (paper/docs/EXPERIMENT_PLAN_SIDE.md Section 16C, written before the runs).

Offline hedging value per model and d in {4, 8} (latency/diagnose_anchor.py --extra, *_extra_v16.json):
    V = (regret_pred - regret_hedge_K4) / regret_naive
Cell mean over the three models.  Closed-loop gain per cell: mean over models of R(hcpp_d k4) - R(hcpp_d k1).
Prediction: Spearman correlation between cell-mean V and cell-mean gain across (cell, d) pairs is
positive; one-sided p by permutation of the gains (100,000 permutations).  Reported for the original
eligible cells (retrodiction), the eligible new cells (prediction made before their closed loop was
read) and all.  Writes results/latency/diag16.json.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency import analyse_hedge_full as AH  # noqa: E402
from latency.analyse_hedge import CELLS as _CELLS  # noqa: E402

OLD = list(_CELLS)
DIAG = ROOT / "results" / "latency" / "diagnose"
P = "pop512_it6"


def V(cell, d):
    f, t = cell.split("/")
    vs = []
    for s in AH.SEEDS:
        p = DIAG / f"{f}_{t}_{s}_extra_v16.json"
        if not p.exists():
            return None
        r = json.load(open(p))["by_d"][str(d)]
        vs.append((r["regret_pred"] - r["regret_hedge_K4"]) / max(r["regret_naive"], 1e-9))
    return float(np.mean(vs))


def gain(cell, d):
    xs = [(AH.load(cell, s, P, f"hcpp{d}k4"), AH.load(cell, s, P, f"hcpp{d}k1")) for s in AH.SEEDS]
    if any(a is None or b is None for a, b in xs):
        return None
    return float(np.mean([a.mean() - b.mean() for a, b in xs]))


def test(cells, rng):
    pts = [(c, d, V(c, d), gain(c, d)) for c in cells for d in (4, 8)]
    pts = [p for p in pts if p[2] is not None and p[3] is not None]
    if len(pts) < 4:
        return {"n": len(pts)}
    v, g = np.array([p[2] for p in pts]), np.array([p[3] for p in pts])
    rho = spearmanr(v, g).statistic
    perm = np.array([spearmanr(v, rng.permutation(g)).statistic for _ in range(100_000)])
    return {"n": len(pts), "spearman": float(rho), "p_one_sided": float((perm >= rho).mean()),
            "points": [{"cell": c, "d": d, "V": vv, "gain": gg} for c, d, vv, gg in pts]}


def main():
    rng = np.random.default_rng(20261018)
    new_path = ROOT / "results" / "latency" / "newtasks.json"
    new = json.load(open(new_path))["eligible_new"] if new_path.exists() else []
    out = {"original": test(OLD, rng), "new": test(new, rng), "all": test(OLD + new, rng)}
    (ROOT / "results" / "latency" / "diag16.json").write_text(json.dumps(out, indent=1))
    for k, r in out.items():
        if "spearman" in r:
            print(f"{k:9s} n = {r['n']:2d}  Spearman {r['spearman']:+.3f}  one-sided p = {r['p_one_sided']:.4f}")
            for p in r["points"]:
                print(f"    {p['cell']:24s} d{p['d']}  V {p['V']:+.3f}  gain {p['gain']:+.3f}")
        else:
            print(k, r)


if __name__ == "__main__":
    main()
