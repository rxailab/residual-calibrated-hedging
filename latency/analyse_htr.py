"""Study 17A analysis (EXPERIMENT_PLAN_SIDE.md Section 17A, written before the runs): hedge-then-refine
on the 45 eligible models (15 cells) of the six tasks.  Confirmatory: R(rcpp_d k4 r1) - R(hcpp_d k4) at
d = 4 and d = 8, pooled with equal cell weight, hierarchical bootstrap, one-sided, Holm over the two
delays.  Secondary: hedge-then-refine vs committed prefix; re-anchored committed prefix vs committed
prefix; hedge-then-refine vs re-anchored committed prefix; mismatch loss and per cell.
Writes results/latency/htr.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency import analyse_hedge_full as AH  # noqa: E402
from latency.analyse_hedge import CELLS as _CELLS  # noqa: E402

OLD = list(_CELLS)
NEW = ["rssm/reacher-easy", "jepa/reacher-easy", "tdmpc2/reacher-easy", "rssm/cheetah-run",
       "rssm/finger-spin", "jepa/finger-spin", "tdmpc2/finger-spin"]
P = "pop512_it6"


def main():
    rng = np.random.default_rng(20261020)
    AH.CELLS[:] = OLD + NEW
    out = {"cells": OLD + NEW, "confirmatory": {}, "secondary": {}, "per_cell": {}}
    for d in (4, 8):
        out["confirmatory"][f"HTR_vs_RCH_d{d}"] = AH.contrast((P, f"rcpp{d}k4r1"), (P, f"hcpp{d}k4"), rng)
        s = out["secondary"]
        s[f"HTR_vs_CPP_d{d}"] = AH.contrast((P, f"rcpp{d}k4r1"), (P, f"hcpp{d}k1"), rng)
        s[f"reCPP_vs_CPP_d{d}"] = AH.contrast((P, f"rcpp{d}k1r1"), (P, f"hcpp{d}k1"), rng)
        s[f"HTR_vs_reCPP_d{d}"] = AH.contrast((P, f"rcpp{d}k4r1"), (P, f"rcpp{d}k1r1"), rng)
        s[f"RCH_vs_CPP_d{d}"] = AH.contrast((P, f"hcpp{d}k4"), (P, f"hcpp{d}k1"), rng)
        for arm in (f"hcpp{d}k1", f"hcpp{d}k4", f"rcpp{d}k1r1", f"rcpp{d}k4r1"):
            s[f"M_{arm}"] = AH.contrast((P, f"paused{d}"), (P, arm), rng)
        for c in OLD + NEW:
            v = {a: float(np.mean([AH.load(c, sd, P, a).mean() for sd in AH.SEEDS]))
                 for a in (f"paused{d}", f"hcpp{d}k1", f"hcpp{d}k4", f"rcpp{d}k1r1", f"rcpp{d}k4r1")}
            out["per_cell"].setdefault(c, {})[f"d{d}"] = v
    conf = out["confirmatory"]
    for k, v in zip(conf, AH.holm([conf[k]["p_one_sided"] for k in conf])):
        conf[k]["p_holm"] = v
    AH.CELLS[:] = OLD
    (ROOT / "results" / "latency" / "htr.json").write_text(json.dumps(out, indent=1))
    for sect in ("confirmatory", "secondary"):
        print(sect)
        for k, v in out[sect].items():
            ph = f", Holm p = {v['p_holm']:.4f}" if "p_holm" in v else ""
            print(f"  {k:18s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  one-sided p = "
                  f"{v['p_one_sided']:.4f}{ph}  ({v['models']} models, {v['cells']} cells)")
    for c, dd in out["per_cell"].items():
        print(f"  {c:24s}", "  ".join(f"d{d[1:]}: CPP {v[f'hcpp{d[1:]}k1']:.3f} RCH {v[f'hcpp{d[1:]}k4']:.3f} "
                                     f"reCPP {v[f'rcpp{d[1:]}k1r1']:.3f} HTR {v[f'rcpp{d[1:]}k4r1']:.3f}"
                                     for d, v in dd.items()))


if __name__ == "__main__":
    main()
