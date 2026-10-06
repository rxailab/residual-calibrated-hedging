"""Collect the offline dataset D for one DMControl task (plan Section 3.2, A9).

Quality spread comes from an epsilon-ladder over the released TD-MPC2 agent:
with probability eps an agent action is replaced by a uniform random action.
Tiers are interleaved by episode index, so every shard is balanced.  Walker's
eps = 0 tier plans with MPPI (its policy prior alone scores ~420); all other
tiers use the policy prior with Gaussian noise sigma = 0.1.

Per step we store 128x128 frames, proprio, qpos/qvel, the reward, and MuJoCo's
full integration state, so any logged moment can be restored exactly with
set_state -- the matched-intervention and plan-cost audits need that.
Everything is a deterministic function of (task, episode index), so the data
lives on scratch and can be regenerated from the pinned agent checkpoints.
"""
import sys, pathlib, time, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from envs.dmc import DMCEnv
from envs.data_agent import TDMPC2Agent, checkpoint_path

EPS_TIERS = [0.0, 0.25, 0.5, 0.75, 1.0]
AGENT_SEED = {"cartpole-swingup": 2, "walker-walk": 1, "cup-catch": 1}
TASK_OFFSET = {"cartpole-swingup": 1_000_000, "walker-walk": 2_000_000, "cup-catch": 3_000_000}
N_EPISODES, EVAL_FROM = 2000, 1900          # episodes 1900-1999 form D_eval (5 %)


def collect(task, start, n, out, img=128, device="cuda"):
    env = DMCEnv(task, img=img)
    obs = env.reset(0)
    kw = dict(obs_dim=obs["proprio"].shape[0], act_dim=env.act_dim, device=device)
    prior = TDMPC2Agent(checkpoint_path(task, AGENT_SEED[task]), mpc=False, **kw)
    expert = TDMPC2Agent(checkpoint_path(task, AGENT_SEED[task]), mpc=True, **kw) \
        if task == "walker-walk" else prior
    buf = {k: [] for k in ("pixels", "proprio", "qpos", "qvel", "mjstate", "action", "reward")}
    meta = {"episode": [], "seed": [], "eps": [], "ret": []}
    t0 = time.time()
    for i in range(start, start + n):
        eps = EPS_TIERS[i % len(EPS_TIERS)]
        agent = expert if eps == 0.0 else prior
        seed = TASK_OFFSET[task] + i
        rng = np.random.default_rng(seed)
        o = env.reset(seed); agent.reset()
        ep = {k: [] for k in buf}
        def log_obs(o):
            ep["pixels"].append(o["pixels"]); ep["proprio"].append(o["proprio"])
            ep["qpos"].append(o["qpos"]); ep["qvel"].append(o["qvel"]); ep["mjstate"].append(env.state())
        log_obs(o)
        while True:
            if rng.random() < eps:
                a = rng.uniform(-1, 1, env.act_dim)
            else:
                a = agent.act(o["proprio"], noise_std=0.1, rng=rng)
            # execute exactly the float32 action that is stored, so replaying a
            # logged action from a logged state reproduces the next state bit-for-bit
            a = np.asarray(a, np.float32)
            o, r, done = env.step(a)
            ep["action"].append(a); ep["reward"].append(r)
            log_obs(o)
            if done:
                break
        for k in buf:
            buf[k].append(np.asarray(ep[k]))
        meta["episode"].append(i); meta["seed"].append(seed); meta["eps"].append(eps)
        meta["ret"].append(float(np.sum(ep["reward"])))
        print(f"  ep {i:4d} eps={eps:.2f} return={meta['ret'][-1]:7.1f} "
              f"steps={len(ep['action'])} ({time.time()-t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.stack(v) for k, v in buf.items()},
                        **{k: np.asarray(v) for k, v in meta.items()},
                        task=task, agent_seed=AGENT_SEED[task], img=img)
    print(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    collect(a.task, a.start, a.n, pathlib.Path(a.out), a.img)
