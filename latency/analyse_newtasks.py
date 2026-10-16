"""Study 16B analysis (paper/docs/EXPERIMENT_PLAN_SIDE.md Section 16B, written before the runs).

New cells: RSSM, JEPA and TD-MPC2 on reacher-easy, cheetah-run and finger-spin, three models each.
A cell is eligible if its mean paused1 normalised return over the three models is at least 0.15
(normalisation from latency/newtask_refs.py).

Confirmatory: R(hcpp_d k4) - R(hcpp_d k1) at d = 4 and d = 8, pooled with equal cell weight over the
eligible new cells; hierarchical bootstrap (latency/analyse_hedge_full.contrast); one-sided, Holm over
the two delays.
Secondary: committed prefix vs time-aligned; mismatch loss paused_d - X_d and the share removed;
per-cell gains; the same contrasts pooled over all eligible cells of the six tasks.
Writes results/latency/newtasks.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency import analyse_hedge_full as AH  # noqa: E402
from latency.analyse import FLOOR  # noqa: E402
from latency.analyse_hedge import CELLS as _CELLS  # noqa: E402

OLD_CELLS = list(_CELLS)         # a copy: run() rewrites the shared list in place

NEW_TASKS = ("reacher-easy", "cheetah-run", "finger-spin")
NEW_CELLS = [f"{f}/{t}" for t in NEW_TASKS for f in ("rssm", "jepa", "tdmpc2")]
P = "pop512_it6"


def mean_level(cell, arm):
    xs = [AH.load(cell, s, P, arm) for s in AH.SEEDS]
    return None if any(x is None for x in xs) else float(np.mean([x.mean() for x in xs]))


def run(cells, rng):
    AH.CELLS[:] = cells          # contrast() iterates over AH.CELLS (imported list)
    out = {}
    for d in (4, 8):
        out[f"K4_vs_K1_d{d}"] = AH.contrast((P, f"hcpp{d}k4"), (P, f"hcpp{d}k1"), rng)
        out[f"K1_vs_aligned_d{d}"] = AH.contrast((P, f"hcpp{d}k1"), (P, f"aligned{d}"), rng)
        out[f"K4_vs_aligned_d{d}"] = AH.contrast((P, f"hcpp{d}k4"), (P, f"aligned{d}"), rng)
        for arm in (f"aligned{d}", f"hcpp{d}k1", f"hcpp{d}k4"):
            out[f"M_{arm}"] = AH.contrast((P, f"paused{d}"), (P, arm), rng)
    return out


def main():
    rng = np.random.default_rng(20261017)
    levels = {c: {a: mean_level(c, a) for a in ("paused1", "paused4", "paused8", "aligned4", "aligned8",
                                                  "hcpp4k1", "hcpp4k4", "hcpp8k1", "hcpp8k4")} for c in NEW_CELLS}
    eligible = [c for c in NEW_CELLS if levels[c]["paused1"] is not None and levels[c]["paused1"] >= FLOOR]
    out = {"levels": levels, "eligible_new": eligible, "floor": FLOOR}
    conf = run(eligible, rng)
    ps = [conf[f"K4_vs_K1_d{d}"]["p_one_sided"] for d in (4, 8)]
    for d, ph in zip((4, 8), AH.holm(ps)):
        conf[f"K4_vs_K1_d{d}"]["p_holm"] = ph
    out["new_cells"] = conf
    out["per_cell_K4_minus_K1"] = {c: {d: levels[c][f"hcpp{d}k4"] - levels[c][f"hcpp{d}k1"] for d in (4, 8)}
                                   for c in eligible}
    out["all_six_tasks"] = run(OLD_CELLS + eligible, rng)
    AH.CELLS[:] = OLD_CELLS
    (ROOT / "results" / "latency" / "newtasks.json").write_text(json.dumps(out, indent=1))
    print("eligible new cells:", eligible)
    for c in NEW_CELLS:
        print(f"  {c:22s}", "  ".join(f"{a} {v:.3f}" for a, v in levels[c].items() if v is not None))
    for sect in ("new_cells", "all_six_tasks"):
        print(sect)
        for k, v in out[sect].items():
            ph = f", Holm p = {v['p_holm']:.4f}" if "p_holm" in v else ""
            print(f"  {k:20s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  one-sided p = "
                  f"{v['p_one_sided']:.4f}{ph}  ({v['models']} models, {v['cells']} cells)")


if __name__ == "__main__":
    main()
