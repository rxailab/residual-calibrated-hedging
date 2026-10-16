"""Harness check for Study 15B (not a paper result): on LeWM's own evaluation protocol
(eval.py at le-wm 8edfeb3: seed 42, 50 episodes, receding horizon 5, CEM batch size 1), the
success of LeWM's reference policy (swm WorldModelPolicy + CEMSolver) against latency/lewm.py's
DelayPolicy with the paused schedule replanning every 5 decisions, on the same episodes.  Also
records, for the paused schedule, at which decision each success happened.
Writes results/latency/lewm/validate_<task>.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import lewm as L  # noqa: E402
import stable_worldmodel as swm  # noqa: E402


def run(task, policy, dataset, eps, starts, n):
    c = L.TASKS[task]
    world = swm.World(**c["world"], num_envs=n, max_episode_steps=2 * L.BUDGET, image_shape=(224, 224))
    world.set_policy(policy)
    m = world.evaluate(dataset=dataset, start_steps=list(map(int, starts)), goal_offset=L.GOAL_OFFSET,
                       eval_budget=L.BUDGET, episodes_idx=list(map(int, eps)), callables=c["callables"])
    world.close()
    return np.asarray(m["episode_successes"], bool)


def main():
    task = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    dev = "cuda"
    c = L.TASKS[task]
    model = L.load_model(task, dev)
    dataset = L.load_dataset(task)
    process = L.processors(dataset, c["keys"])
    eps, starts = L.eval_starts(dataset, n, 42)
    tfm = {"pixels": L.img_transform(), "goal": L.img_transform()}
    out = {"task": task, "episodes": n, "seed": 42}
    t0 = time.time()
    solver = swm.solver.CEMSolver(model=model, batch_size=1, num_samples=300, var_scale=1.0, n_steps=30, topk=30,
                                  device=dev, seed=42)
    cfg = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    ref = swm.policy.WorldModelPolicy(solver=solver, config=cfg, process=process, transform=tfm)
    s_ref = run(task, ref, dataset, eps, starts, n)
    out["reference_success"] = float(s_ref.mean()); out["reference_s"] = time.time() - t0
    for arm in ("paused5", "paused1"):
        t0 = time.time()
        pol = L.DelayPolicy(model, arm, process, tfm, seed=777, dev=dev)
        s = run(task, pol, dataset, eps, starts, n)
        out[f"{arm}_success"] = float(s.mean()); out[f"{arm}_s"] = time.time() - t0
        out[f"{arm}_call_s_mean"] = float(np.mean(pol.call_s))
    print(json.dumps(out, indent=1), flush=True)
    od = L.OUT; od.mkdir(parents=True, exist_ok=True)
    (od / f"validate_{task}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
