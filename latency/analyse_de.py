"""Confirmatory analysis of the delay-error method (sidepaper/EXPERIMENT_PLAN_SIDE.md, Section 13).

Contrasts on the seed1/seed2 models (16 models in 8 cells), d = 4 and 8, one-sided,
Holm over all eight, hierarchical bootstrap (models within cell, episodes within
model; 10,000 resamples):
  C1 ecpp k4 - hcpp k4          learned vs unconditional hedge
  C2 ecpp k4 h* - ecpp k4       hand-off penalty adds return
  C3 ecpp k4 h* - ecpp k4 r*    hand-off penalty beats a generic risk penalty
  C4 ecpp k4 h* - hcpp k1       full method vs plain committed-prefix planning
Secondary: mcpp and scpp vs ecpp k4 and hcpp k1; ecpp k4 r* vs hcpp k1 and ecpp k4.
Writes results/latency/de_confirm.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency.analyse_hedge_full import holm, load  # noqa: E402
from latency.analyse_hedge import CELLS  # noqa: E402
from paths import RESULTS  # noqa: E402

SEEDS = ("seed1", "seed2")
B = 10_000
P5 = "pop512_it6"


def contrast(a, b, rng):
    cells, n = [], 0
    for c in CELLS:
        ms = []
        for s in SEEDS:
            x, y = load(c, s, P5, a), load(c, s, P5, b)
            if x is not None and y is not None:
                ms.append(x - y)
        if ms:
            cells.append(np.stack(ms)); n += len(ms)
    if not cells:
        return None
    est = float(np.mean([x.mean() for x in cells]))
    bs = np.zeros(B)
    for x in cells:
        m, k = x.shape
        mi = rng.integers(0, m, (B, m)); ei = rng.integers(0, k, (B, m, k))
        bs += x[mi[..., None], ei].mean((1, 2))
    bs /= len(cells)
    return {"est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_one_sided": float((bs <= 0).mean()), "models": n, "cells": len(cells)}


def main():
    sel = json.loads((RESULTS / "latency" / "tune_selection.json").read_text())
    lh, lr = sel["lam_h"], sel["lam_r"]
    rng = np.random.default_rng(20261006)
    prim = {}
    for d in (4, 8):
        prim[f"C1_learned_vs_uncond_d{d}"] = contrast(f"ecpp{d}k4", f"hcpp{d}k4", rng)
        prim[f"C2_handoff_vs_none_d{d}"] = contrast(f"ecpp{d}k4h{lh}", f"ecpp{d}k4", rng)
        prim[f"C3_handoff_vs_risk_d{d}"] = contrast(f"ecpp{d}k4h{lh}", f"ecpp{d}k4r{lr}", rng)
        prim[f"C4_full_vs_cpp_d{d}"] = contrast(f"ecpp{d}k4h{lh}", f"hcpp{d}k1", rng)
    keys = [k for k, v in prim.items() if v]
    for k, p in zip(keys, holm([prim[k]["p_one_sided"] for k in keys])):
        prim[k]["p_holm"] = p
    sec = {}
    for d in (4, 8):
        sec[f"ecpp_vs_cpp_d{d}"] = contrast(f"ecpp{d}k4", f"hcpp{d}k1", rng)
        sec[f"risk_vs_cpp_d{d}"] = contrast(f"ecpp{d}k4r{lr}", f"hcpp{d}k1", rng)
        sec[f"risk_vs_ecpp_d{d}"] = contrast(f"ecpp{d}k4r{lr}", f"ecpp{d}k4", rng)
        sec[f"mcpp_vs_cpp_d{d}"] = contrast(f"mcpp{d}", f"hcpp{d}k1", rng)
        sec[f"scpp_vs_ecpp_d{d}"] = contrast(f"scpp{d}k4", f"ecpp{d}k4", rng)
        sec[f"uncond_vs_cpp_d{d}"] = contrast(f"hcpp{d}k4", f"hcpp{d}k1", rng)
        sec[f"ecpp_vs_aligned_d{d}"] = contrast(f"ecpp{d}k4", f"aligned{d}", rng)
    out = {"lam_h": lh, "lam_r": lr, "primary": prim, "secondary": sec}
    (RESULTS / "latency" / "de_confirm.json").write_text(json.dumps(out, indent=1))
    for name, dd in (("primary", prim), ("secondary", sec)):
        print(name)
        for k, v in dd.items():
            if v:
                ph = f", Holm p = {v['p_holm']:.4f}" if "p_holm" in v else ""
                print(f"  {k:28s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  p = {v['p_one_sided']:.4f}{ph}"
                      f"  ({v['models']} models)")


if __name__ == "__main__":
    main()
