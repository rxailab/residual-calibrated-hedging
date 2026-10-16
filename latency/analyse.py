"""Side-paper analysis (sidepaper/EXPERIMENT_PLAN_SIDE.md, Sections 7-8; Supplementary Note 1).

Written on 5 October 2026 after the Study 1-3 runs had finished but before any of
their results were read; it implements the estimands fixed in the plan unchanged.

Normalised return (R - R_rand) / (R_exp - R_rand) per episode.  A (family, task)
cell enters pooled estimates if its mean paused1 return over the three models
is at least 0.15.  Pooled estimates weight cells equally.  Intervals: 95 %
percentile, hierarchical bootstrap (models within cell, then episodes within
model, the same episode indices for every arm of a model), 10,000 resamples.

  H1  R(cpp_d) - R(aligned_d) > 0 at each d in {2, 4, 8} (intersection-union:
      p = the largest of the three one-sided bootstrap p-values).
  H2  M_cpp(d) <= 0.5 M_aligned(d) at d in {4, 8}, M_X(d) = R(paused_d) - R(X_d)
      (p = the largest one-sided bootstrap p-value of ratio > 0.5).
  H3  needs the latency measurements (Study 2 timing); reported when they exist.
Holm over H1-H3 when all three p-values exist.

Writes results/latency/summary.json; prints the tables.
"""
import glob
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paths import RESULTS  # noqa: E402

REF = {"cartpole-swingup": (880, 49), "walker-walk": (985, 40), "cup-catch": (981, 55)}
_NEW = ROOT / "results" / "latency" / "newtask_refs.json"          # Section 16B (latency/newtask_refs.py)
if _NEW.exists():
    REF.update({k: tuple(v) for k, v in json.load(open(_NEW)).items()})
FAMILIES = ("rssm", "jepa", "tdmpc2")
TASKS = tuple(REF)
DS = (1, 2, 4, 8, 16)
B = 10_000
FLOOR = 0.15
RNG = np.random.default_rng(20261005)


def norm(task, x):
    e, r = REF[task]
    return (np.asarray(x, float) - r) / (e - r)


def load_dmc(config="pop512_it6", runs=("seed0", "seed1", "seed2")):
    """cells[(family, task)][arm] = (n_models, n_episodes) normalised returns."""
    cells = {}
    for f in FAMILIES:
        for t in TASKS:
            arms = {}
            for run in runs:
                for p in glob.glob(str(RESULTS / "latency" / "dmc" / f / t / run / config / "*.json")):
                    d = json.load(open(p))
                    arms.setdefault(d["arm"], {})[run] = norm(t, d["returns"])
            full = {a: np.stack([v[r] for r in runs]) for a, v in arms.items() if all(r in v for r in runs)}
            if full:
                cells[(f, t)] = full
    return cells


def boot_cell(R, B, rng):
    """Bootstrap index arrays for one cell: models (B, m) and episodes (B, m, n)."""
    m, n = next(iter(R.values())).shape
    mi = rng.integers(0, m, (B, m))
    ei = rng.integers(0, n, (B, m, n))
    return mi, ei


def cell_means(R, mi, ei):
    return {a: x[mi[..., None], ei].mean((1, 2)) for a, x in R.items()}


def pct(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    cells = load_dmc()
    out = {"cells": {}, "pooled": {}, "hypotheses": {}}
    eligible = []
    for (f, t), R in sorted(cells.items()):
        p1 = float(R["paused1"].mean()) if "paused1" in R else float("nan")
        ok = p1 >= FLOOR
        if ok:
            eligible.append((f, t))
        out["cells"][f"{f}/{t}"] = {"paused1": p1, "eligible": ok,
                                    "arms": {a: float(x.mean()) for a, x in sorted(R.items())}}
    # pooled bootstrap over eligible cells
    boots = {}
    for c in eligible:
        mi, ei = boot_cell(cells[c], B, RNG)
        boots[c] = cell_means(cells[c], mi, ei)
    point = {c: {a: float(x.mean()) for a, x in cells[c].items()} for c in eligible}

    def pooled(fn):
        est = float(np.mean([fn(point[c]) for c in eligible]))
        bs = np.mean([fn(boots[c]) for c in eligible], 0)
        return est, bs

    for d in DS:
        pd_ = "paused1" if d == 1 else f"paused{d}"
        row = {}
        for name, fn in {
            "R_paused1": lambda m: m["paused1"],
            "R_paused_d": lambda m: m[pd_],
            "R_naive": lambda m: m[f"naive{d}"],
            "R_aligned": lambda m: m[f"aligned{d}"],
            "R_cpp": lambda m: m[f"cpp{d}"],
            "cpp_minus_aligned": lambda m: m[f"cpp{d}"] - m[f"aligned{d}"],
            "F": lambda m: m["paused1"] - m[pd_],
            "M_aligned": lambda m: m[pd_] - m[f"aligned{d}"],
            "M_cpp": lambda m: m[pd_] - m[f"cpp{d}"],
            "M_naive": lambda m: m[pd_] - m[f"naive{d}"],
        }.items():
            est, bs = pooled(fn)
            row[name] = {"est": est, "ci95": pct(bs)}
            if name == "cpp_minus_aligned":
                row[name]["p_one_sided"] = float((bs <= 0).mean())
        ea, ba = pooled(lambda m: m[pd_] - m[f"aligned{d}"])
        ec, bc = pooled(lambda m: m[pd_] - m[f"cpp{d}"])
        ratio_b = bc / np.where(np.abs(ba) > 1e-12, ba, np.nan)
        row["ratio_M_cpp_over_M_aligned"] = {"est": ec / ea if abs(ea) > 1e-12 else float("nan"),
                                             "ci95": [float(np.nanpercentile(ratio_b, 2.5)),
                                                      float(np.nanpercentile(ratio_b, 97.5))],
                                             "p_ratio_gt_half": float(np.nanmean(ratio_b > 0.5))}
        out["pooled"][f"d{d}"] = row
    p1 = max(out["pooled"][f"d{d}"]["cpp_minus_aligned"]["p_one_sided"] for d in (2, 4, 8))
    p2 = max(out["pooled"][f"d{d}"]["ratio_M_cpp_over_M_aligned"]["p_ratio_gt_half"] for d in (4, 8))
    out["hypotheses"] = {"H1": {"p": p1}, "H2": {"p": p2}, "H3": {"p": None, "note": "needs Study 2 timing"},
                         "eligible_cells": [f"{f}/{t}" for f, t in eligible], "B": B, "floor": FLOOR}
    # diffusion (descriptive)
    diff = {}
    for t in ("cartpole-swingup", "cup-catch"):
        arms = {}
        for p in glob.glob(str(RESULTS / "latency" / "diffusion" / t / "*" / "s2" / "*.json")):
            d = json.load(open(p))
            arms.setdefault(d["arm"], []).extend(norm(t, d["returns"]).tolist())
        diff[t] = {a: {"mean": float(np.mean(v)), "n": len(v),
                       "se": float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")}
                   for a, v in sorted(arms.items())}
    out["diffusion"] = diff
    od = RESULTS / "latency"
    (od / "summary.json").write_text(json.dumps(out, indent=1))

    # ---- print
    print("Cells (mean normalised return over 3 models x 50 episodes; * = below the 0.15 floor)")
    hdr = ["paused1"] + [f"{k}{d}" for d in (2, 4, 8, 16) for k in ("paused", "aligned", "cpp")]
    print(f"{'cell':28s}" + "".join(f"{h:>10s}" for h in hdr))
    for c, v in out["cells"].items():
        print(f"{c + ('' if v['eligible'] else ' *'):28s}" + "".join(f"{v['arms'].get(h, float('nan')):10.3f}" for h in hdr))
    print(f"\nPooled over {len(eligible)} eligible cells")
    print(f"{'d':>3s} {'paused1':>8s} {'paused_d':>9s} {'naive':>7s} {'aligned':>8s} {'cpp':>7s} "
          f"{'cpp-aligned [95% CI]':>26s} {'F(d)':>7s} {'M_aligned':>10s} {'M_cpp':>7s} {'ratio [CI]':>20s}")
    for d in DS:
        r = out["pooled"][f"d{d}"]
        ca = r["cpp_minus_aligned"]; ra = r["ratio_M_cpp_over_M_aligned"]
        print(f"{d:3d} {r['R_paused1']['est']:8.3f} {r['R_paused_d']['est']:9.3f} {r['R_naive']['est']:7.3f} "
              f"{r['R_aligned']['est']:8.3f} {r['R_cpp']['est']:7.3f} "
              f"{ca['est']:+8.3f} [{ca['ci95'][0]:+.3f}, {ca['ci95'][1]:+.3f}] {r['F']['est']:7.3f} "
              f"{r['M_aligned']['est']:10.3f} {r['M_cpp']['est']:7.3f} "
              f"{ra['est']:6.2f} [{ra['ci95'][0]:.2f}, {ra['ci95'][1]:.2f}]")
    print(f"\nH1 p = {p1:.4f} (max over d = 2, 4, 8);  H2 p = {p2:.4f} (max over d = 4, 8);  H3 pending timing")
    print("\nDiffusion (normalised return, 10 episodes per arm)")
    for t, v in diff.items():
        print(" ", t, "  ".join(f"{a} {x['mean']:+.3f}±{x['se']:.3f}" for a, x in v.items()))


if __name__ == "__main__":
    main()
