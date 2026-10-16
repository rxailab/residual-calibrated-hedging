"""Side paper Section 16B, before data collection: the released TD-MPC2 agent's return on a new
task with its policy prior and with MPC (seed 1, 3 episodes each, action repeat 2, 500 decisions).
Rule fixed before the check: collect the epsilon = 0 tier with MPC if the prior's mean return is
below 0.85 x MPC's, otherwise with the prior (as for cartpole and cup catch).
Writes results/latency/agents/<task>.json."""
import json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from envs.dmc import DMCEnv
from envs.data_agent import TDMPC2Agent, checkpoint_path
from paths import RESULTS

task = sys.argv[1]
env = DMCEnv(task, render=False)
obs = env.reset(0)
kw = dict(obs_dim=obs["proprio"].shape[0], act_dim=env.act_dim, device="cuda")
out = {"task": task, "seed": 1}
for name, mpc in (("prior", False), ("mpc", True)):
    agent = TDMPC2Agent(checkpoint_path(task, 1), mpc=mpc, **kw)
    rets, t0 = [], time.time()
    for ep in range(3):
        o = env.reset(7_000_000 + ep); agent.reset(); R = 0.0
        while True:
            o, r, done = env.step(agent.act(o["proprio"]))
            R += r
            if done:
                break
        rets.append(R)
    out[name] = {"returns": rets, "mean": float(np.mean(rets)), "seconds": time.time() - t0}
    print(task, name, [round(x, 1) for x in rets], flush=True)
out["use_mpc"] = bool(out["prior"]["mean"] < 0.85 * out["mpc"]["mean"])
print(task, "use_mpc", out["use_mpc"], flush=True)
od = RESULTS / "latency" / "agents"; od.mkdir(parents=True, exist_ok=True)
(od / f"{task}.json").write_text(json.dumps(out, indent=1))
