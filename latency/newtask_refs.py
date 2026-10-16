"""Section 16B normalisation references for the new tasks, fixed before the runs: R_exp = mean
return of the epsilon = 0 tier of the collected data (the released TD-MPC2 agent), R_rand = mean
return of the epsilon = 1 tier (uniform random actions), over all 2,000 episodes.
Reads $RCH_SCRATCH/data/dmc64/<task>/meta.npz; writes results/latency/newtask_refs.json."""
import json, os, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCR = pathlib.Path(os.environ.get("RCH_SCRATCH", "/scratch/hpc/11/xiar3/RCH"))
out = {}
for t in ("reacher-easy", "cheetah-run", "finger-spin"):
    m = np.load(SCR / "data" / "dmc64" / t / "meta.npz")
    eps, ret = m["eps"], m["ret"]
    out[t] = [float(ret[eps == 0.0].mean()), float(ret[eps == 1.0].mean())]
    print(t, "R_exp %.1f  R_rand %.1f" % tuple(out[t]), {float(e): round(float(ret[eps == e].mean()), 1) for e in np.unique(eps)})
(ROOT / "results" / "latency" / "newtask_refs.json").write_text(json.dumps(out, indent=1))
