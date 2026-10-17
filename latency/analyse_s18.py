"""Section 18 analysis (EXPERIMENT_PLAN_SIDE.md, written before the runs).
18A (confirmatory): released TD-MPC2, new episodes (seeds 950,100+): R(rchl_d k4) - R(cpp_d) at d = 4
and 8, equal task weight, hierarchical bootstrap, one-sided, Holm over two; secondary rchl vs rch.
18B (secondary): in-house pixel TD-MPC2 cells: hcpp{d}k4l vs hcpp{d}k1 and vs hcpp{d}k4.
Writes results/latency/s18.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency import analyse_tdmpc2_released as TR  # noqa: E402
from latency import analyse_hedge_full as AH  # noqa: E402
from latency.analyse_hedge import CELLS as _CELLS  # noqa: E402

OLD = list(_CELLS)
P = "pop512_it6"


def main():
    rng = np.random.default_rng(20261021)
    TR.D = ROOT / "results" / "latency" / "tdmpc2_released_s18"
    out = {"18A": {}, "18A_secondary": {}, "18B": {}}
    for d in (4, 8):
        out["18A"][f"rchl_vs_cpp_d{d}"] = TR.contrast(TR.TASKS, f"rchl{d}k4", f"cpp{d}", rng)
        out["18A_secondary"][f"rchl_vs_rch_d{d}"] = TR.contrast(TR.TASKS, f"rchl{d}k4", f"rch{d}k4", rng)
        out["18A_secondary"][f"rch_vs_cpp_d{d}"] = TR.contrast(TR.TASKS, f"rch{d}k4", f"cpp{d}", rng)
        out["18A_secondary"][f"M_cpp{d}"] = TR.contrast(TR.TASKS, "paused1", f"cpp{d}", rng)
        out["18A_secondary"][f"M_rchl{d}"] = TR.contrast(TR.TASKS, "paused1", f"rchl{d}k4", rng)
    for k, v in zip(out["18A"], AH.holm([out["18A"][k]["p_one_sided"] for k in out["18A"]])):
        out["18A"][k]["p_holm"] = v
    out["18A_levels"] = {t: {a: (float(v.mean()) if (v := TR.load(t, a)) is not None else None)
                             for a in ("paused1", "cpp4", "rch4k4", "rchl4k4", "cpp8", "rch8k4", "rchl8k4")} for t in TR.TASKS}
    AH.CELLS[:] = ["tdmpc2/cartpole-swingup", "tdmpc2/cup-catch", "tdmpc2/walker-walk",
                   "tdmpc2/reacher-easy", "tdmpc2/finger-spin"]
    for d in (4, 8):
        out["18B"][f"log_vs_CPP_d{d}"] = AH.contrast((P, f"hcpp{d}k4l"), (P, f"hcpp{d}k1"), rng)
        out["18B"][f"log_vs_additive_d{d}"] = AH.contrast((P, f"hcpp{d}k4l"), (P, f"hcpp{d}k4"), rng)
        out["18B"][f"additive_vs_CPP_d{d}"] = AH.contrast((P, f"hcpp{d}k4"), (P, f"hcpp{d}k1"), rng)
    AH.CELLS[:] = OLD
    (ROOT / "results" / "latency" / "s18.json").write_text(json.dumps(out, indent=1))
    for t, lv in out["18A_levels"].items():
        print(f"  {t:17s}", "  ".join(f"{a} {v:.3f}" for a, v in lv.items() if v is not None))
    for sect in ("18A", "18A_secondary", "18B"):
        print(sect)
        for k, v in out[sect].items():
            ph = f", Holm p = {v['p_holm']:.4f}" if "p_holm" in v else ""
            print(f"  {k:22s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  one-sided p = {v['p_one_sided']:.4f}{ph}")


if __name__ == "__main__":
    main()
