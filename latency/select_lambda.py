"""Apply the pre-registered tuning rule (sidepaper/EXPERIMENT_PLAN_SIDE.md, Section 13)
and write the confirmatory run list.

lam_h = argmax over {0.25, 0.5, 1.0} of the pooled tuning return of ecpp<d>k4h<lam>,
lam_r likewise for ecpp<d>k4r<lam>; pooled = mean over the 8 seed0 cells and d in {4, 8}
of the mean normalised return over the 20 tuning episodes (seeds 970,000-970,019).
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

LAMS = ("0.25", "0.5", "1.0")
TUNE = RESULTS / "latency" / "tune"
CK = ROOT / "checkpoints_keep" / "dmc"


def pooled(arm_fmt):
    vals = []
    for c in CELLS:
        for d in (4, 8):
            f = TUNE / c / "seed0" / "pop512_it6" / f"{arm_fmt.format(d=d)}.json"
            vals.append(norm(c.split("/")[1], json.load(open(f))["returns"]).mean())
    return float(np.mean(vals))


def main():
    base = pooled("ecpp{d}k4")
    sh = {l: pooled("ecpp{d}k4h" + l) for l in LAMS}
    sr = {l: pooled("ecpp{d}k4r" + l) for l in LAMS}
    lh, lr = max(sh, key=sh.get), max(sr, key=sr.get)
    out = {"ecpp_k4": base, "hand_off": sh, "risk": sr, "lam_h": lh, "lam_r": lr}
    (RESULTS / "latency" / "tune_selection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    lines = []
    for s in ("seed1", "seed2"):
        for c in CELLS:
            fam = c.split("/")[0]; st = "0100000" if fam == "tdmpc2" else "0060000"
            arms = []
            for d in (8, 4):
                arms += [f"ecpp{d}k4", f"ecpp{d}k4h{lh}", f"ecpp{d}k4r{lr}", f"mcpp{d}", f"scpp{d}k4"]
            lines.append(f"{CK}/{c}/{s}/step{st}.pt --arms {','.join(arms)}")
    (ROOT / "latency" / "runs_de_confirm.txt").write_text("\n".join(lines) + "\n")
    print("wrote", len(lines), "confirmatory lines")


if __name__ == "__main__":
    main()
