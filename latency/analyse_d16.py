"""Study 15A (paper/docs/EXPERIMENT_PLAN_SIDE.md Section 15, written before the runs):
R(K=4) - R(K=1) at d = 16 over the 24 models, pooled with equal cell weight, hierarchical
bootstrap (latency/analyse_hedge_full.contrast), one-sided, alpha = 0.05, a single test.
Secondary: mismatch loss against paused16, and the K = 4 gain per cell.
Writes results/latency/d16.json.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latency.analyse_hedge import CELLS  # noqa: E402
from latency.analyse_hedge_full import contrast, load, SEEDS  # noqa: E402
from paths import RESULTS  # noqa: E402

P = "pop512_it6"


def main():
    rng = np.random.default_rng(20261016)
    out = {"confirmatory_K4_vs_K1_d16": contrast((P, "hcpp16k4"), (P, "hcpp16k1"), rng)}
    out["secondary"] = {"K1_vs_aligned_d16": contrast((P, "hcpp16k1"), (P, "aligned16"), rng),
                        "K4_vs_aligned_d16": contrast((P, "hcpp16k4"), (P, "aligned16"), rng),
                        "M_K1_d16": contrast((P, "paused16"), (P, "hcpp16k1"), rng),
                        "M_K4_d16": contrast((P, "paused16"), (P, "hcpp16k4"), rng)}
    out["per_cell"] = {c: float(np.mean([load(c, s, P, "hcpp16k4").mean() - load(c, s, P, "hcpp16k1").mean()
                                         for s in SEEDS])) for c in CELLS}
    (RESULTS / "latency" / "d16.json").write_text(json.dumps(out, indent=1))
    for k, v in [("confirmatory", out["confirmatory_K4_vs_K1_d16"])] + list(out["secondary"].items()):
        print(f"{k:22s} {v['est']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  one-sided p = {v['p_one_sided']:.4f}")
    print({c.split("/")[0][:4] + "-" + c.split("/")[1][:4]: round(v, 3) for c, v in out["per_cell"].items()})


if __name__ == "__main__":
    main()
