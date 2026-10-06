"""Confirmatory analysis of residual-hedged planning over all 24 models
(sidepaper/EXPERIMENT_PLAN_SIDE.md, Section 12; Supplementary Note 1), with the
rules written before the replication and equal-latency runs:

  replication   R(K16, pop512) - R(K1, pop512) at d = 4 and d = 8, one-sided,
                Holm over the two delays.  Secondary: d = 2; K4 vs K1.
  equal latency R(K4, pop512) - R(K1, pop512) and R(K16, pop128) - R(K1, pop512)
                at d = 4 and d = 8, one-sided, Holm over the four contrasts.

Pooled with equal cell weight over the 8 cells; hierarchical bootstrap (models
within cell, then episodes within model, shared indices across the two arms of
a contrast); 10,000 resamples.  One-sided p = share of bootstrap means <= 0.
Writes results/latency/hedge_full.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency.analyse import norm  # noqa: E402
from latency.analyse_hedge import CELLS  # noqa: E402
from paths import RESULTS  # noqa: E402

SEEDS = ("seed0", "seed1", "seed2")
B = 10_000


def load(cell, seed, cfg, arm):
    f = RESULTS / "latency" / "dmc" / cell / seed / cfg / f"{arm}.json"
    return norm(cell.split("/")[1], json.load(open(f))["returns"]) if f.exists() else None


def contrast(a, b, rng):
    """a, b: (cfg, arm) pairs. Returns estimate, 95 % CI, one-sided p, models used."""
    cells, n_models = [], 0
    for c in CELLS:
        ms = []
        for s in SEEDS:
            x, y = load(c, s, *a), load(c, s, *b)
            if x is not None and y is not None:
                ms.append(x - y)
        if ms:
            cells.append(np.stack(ms)); n_models += len(ms)
    est = float(np.mean([x.mean() for x in cells]))
    bs = np.zeros(B)
    for x in cells:
        m, n = x.shape
        mi = rng.integers(0, m, (B, m)); ei = rng.integers(0, n, (B, m, n))
        bs += x[mi[..., None], ei].mean((1, 2))
    bs /= len(cells)
    return {"est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_one_sided": float((bs <= 0).mean()), "models": n_models, "cells": len(cells)}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj.tolist()


def main():
    rng = np.random.default_rng(20261005)
    P5, P1 = "pop512_it6", "pop128_it6"
    out = {}
    rep = {f"d{d}": contrast((P5, f"hcpp{d}k16"), (P5, f"hcpp{d}k1"), rng) for d in (4, 8)}
    for k, v in zip(rep, holm([rep[k]["p_one_sided"] for k in rep])):
        rep[k]["p_holm"] = v
    out["replication_K16_vs_K1"] = rep
    out["secondary"] = {"K16_vs_K1_d2": contrast((P5, "hcpp2k16"), (P5, "hcpp2k1"), rng),
                        **{f"K4_vs_K1_d{d}": contrast((P5, f"hcpp{d}k4"), (P5, f"hcpp{d}k1"), rng) for d in (2, 4, 8)},
                        **{f"K1_vs_aligned_d{d}": contrast((P5, f"hcpp{d}k1"), (P5, f"aligned{d}"), rng) for d in (2, 4, 8)},
                        **{f"K4_vs_aligned_d{d}": contrast((P5, f"hcpp{d}k4"), (P5, f"aligned{d}"), rng) for d in (2, 4, 8)}}
    eq = {}
    for d in (4, 8):
        eq[f"K4p512_vs_K1p512_d{d}"] = contrast((P5, f"hcpp{d}k4"), (P5, f"hcpp{d}k1"), rng)
        eq[f"K16p128_vs_K1p512_d{d}"] = contrast((P1, f"hcpp{d}k16"), (P5, f"hcpp{d}k1"), rng)
    for k, v in zip(eq, holm([eq[k]["p_one_sided"] for k in eq])):
        eq[k]["p_holm"] = v
    out["equal_latency"] = eq
    (RESULTS / "latency" / "hedge_full.json").write_text(json.dumps(out, indent=1))
    for sect, dd in out.items():
        print(sect)
        for k, v in dd.items():
            ph = f", Holm p = {v['p_holm']:.4f}" if "p_holm" in v else ""
            print(f"  {k:24s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  one-sided p = "
                  f"{v['p_one_sided']:.4f}{ph}  ({v['models']} models, {v['cells']} cells)")


if __name__ == "__main__":
    main()
