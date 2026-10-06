"""Hedged-planning pilot (sidepaper/EXPERIMENT_PLAN_SIDE.md, Section 12): the
decision rule written before the runs.

Primary contrast: R(hcpp_d k16) - R(hcpp_d k1), normalised return, pooled with
equal weight over the 8 eligible seed0 cells, at d = 2, 4, 8; 95 % percentile
interval from an episode bootstrap (same episode indices for both arms of a
cell; 10,000 resamples).  Proceed if the interval is above 0 at >= 2 delays;
fall back if it includes 0 at all three.  Also reported: the fraction of the
remaining mismatch removed, (R(k16) - R(k1)) / (R(paused_d) - R(k1)).
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency.analyse import REF, norm  # noqa: E402
from paths import RESULTS  # noqa: E402

CELLS = ["jepa/cartpole-swingup", "jepa/cup-catch", "rssm/cartpole-swingup", "rssm/cup-catch",
         "rssm/walker-walk", "tdmpc2/cartpole-swingup", "tdmpc2/cup-catch", "tdmpc2/walker-walk"]
B = 10_000


def load(cell, arm):
    f = RESULTS / "latency" / "dmc" / cell / "seed0" / "pop512_it6" / f"{arm}.json"
    if not f.exists():
        return None
    return norm(cell.split("/")[1], json.load(open(f))["returns"])


def main():
    rng = np.random.default_rng(20261005)
    out = {}
    print(f"{'cell':26s} {'d':>2s} {'paused_d':>8s} {'aligned':>8s} {'cpp':>7s} {'k1':>7s} {'k4':>7s} {'k16':>7s}")
    for d in (2, 4, 8):
        diffs, fr_num, fr_den, rows = [], [], [], {}
        for c in CELLS:
            arms = {a: load(c, a) for a in (f"paused{d}", f"aligned{d}", f"cpp{d}",
                                             f"hcpp{d}k1", f"hcpp{d}k4", f"hcpp{d}k16")}
            if arms[f"hcpp{d}k1"] is None or arms[f"hcpp{d}k16"] is None:
                continue
            m = {k: (float(v.mean()) if v is not None else float("nan")) for k, v in arms.items()}
            rows[c] = m
            print(f"{c:26s} {d:2d} " + " ".join(f"{m[k]:8.3f}" for k in arms))
            diffs.append(arms[f"hcpp{d}k16"] - arms[f"hcpp{d}k1"])
            fr_num.append(m[f"hcpp{d}k16"] - m[f"hcpp{d}k1"]); fr_den.append(m[f"paused{d}"] - m[f"hcpp{d}k1"])
        if not diffs:
            continue
        D = np.stack(diffs)                                    # (cells, episodes), paired
        n = D.shape[1]
        idx = rng.integers(0, n, (B, n))
        boot = np.stack([D[i][idx].mean(1) for i in range(len(D))]).mean(0)
        est = float(D.mean(1).mean())
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        frac = float(np.sum(fr_num) / np.sum(fr_den)) if np.sum(fr_den) > 0 else float("nan")
        out[f"d{d}"] = {"cells": rows, "k16_minus_k1": {"est": est, "ci95": ci}, "n_cells": len(D),
                        "fraction_of_remaining_mismatch_removed": frac}
        print(f"   d={d}: k16 - k1 = {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] over {len(D)} cells; "
              f"fraction of remaining mismatch removed {frac:.2f}\n")
    n_pos = sum(v["k16_minus_k1"]["ci95"][0] > 0 for v in out.values())
    n_zero = sum(v["k16_minus_k1"]["ci95"][0] <= 0 <= v["k16_minus_k1"]["ci95"][1] for v in out.values())
    verdict = "proceed" if n_pos >= 2 else ("fall back" if n_zero == len(out) == 3 else "discuss")
    out["verdict"] = verdict
    print("decision rule:", verdict, f"(interval above 0 at {n_pos} of 3 delays)")
    (RESULTS / "latency" / "hedge_pilot.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
