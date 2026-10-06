"""Probe and reward head of the diffusion family, per task (plan Appendix B and
Section 4, "Reward heads").

The diffusion family's latent is the frozen Wan2.2 VAE latent (one 48x8x8
frame per block of 4 decisions, wm/diffusion.py), which is the same for every
checkpoint.  The probe and the reward head are therefore fit once per task on
D_train latents and shared by all checkpoints: "refit per checkpoint with an
identical recipe" would refit the same function on the same inputs.  Every
difference between diffusion checkpoints is then a difference in dynamics.

  probe   ridge from latent k (3072 numbers) to the z-scored concat(proprio,
          state) at frame 4k, penalty by 5-fold CV holding out whole episodes
          (amendment A12's grouping, at episode level)
  reward  MLP 3072-512-512-1 on latent k, k >= 1, predicting symlog of the
          block reward (the sum of the 4 rewards of decisions 4k-4 .. 4k-1),
          MSE, Adam 3e-4, batch 1024, 20k updates; the RSSM and JEPA heads also
          regress symlog rewards with MSE
  kin4    constant-velocity null model over one block for M12:
          qpos_{t+4} = qpos_t + c4 * qvel_t, least squares per joint on D_train
  var     total variance of D_eval latents, for the normalised latent error

Writes results/diffusion/heads/<task>.pt.
"""
import sys, pathlib, json, time, argparse
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from experiments.dmc.evaluate import task_data, ALPHAS, N_TRAIN, group_r2
from wm.rssm import mlp, symlog, symexp
from paths import RESULTS

LAT = pathlib.Path("/scratch/hpc/11/xiar3/MetricTrust/data/dmc_lat")
OUT = RESULTS / "diffusion" / "heads"
N_LAT, BLOCK = 126, 4


def load_latents(task, shards=range(40)):
    return np.concatenate([np.load(LAT / task / f"lat_{k:03d}.npy") for k in shards])


def block_targets(td, eps):
    """z-scored concat(proprio, state) at frames 0, 4, ..., 500 of episodes eps -> (E, 126, n)."""
    return np.stack([td.targets(e, 0, 501)[::BLOCK] for e in eps])


def block_rewards(reward):
    """(E, 500) -> (E, 125) sums over decisions 4k-4 .. 4k-1, k = 1 .. 125."""
    return reward.reshape(reward.shape[0], -1, BLOCK).sum(-1)


class RewardHead(nn.Module):
    def __init__(self, d=48 * 64, hidden=512):
        super().__init__()
        self.net = mlp(d, hidden, 1)

    def forward(self, z):
        return self.net(z.reshape(*z.shape[:-3], -1) if z.dim() >= 4 else z).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="cartpole-swingup,walker-walk,cup-catch")
    ap.add_argument("--probe-eps", type=int, default=300)
    ap.add_argument("--steps", type=int, default=20000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for task in a.tasks.split(","):
        t0 = time.time()
        td = task_data(task)
        lat = load_latents(task).astype(np.float32)                   # (2000, 126, 48, 8, 8)
        D = int(np.prod(lat.shape[2:]))
        m = np.load(pathlib.Path("/scratch/hpc/11/xiar3/MetricTrust/data/dmc64") / task / "meta.npz")
        rew = block_rewards(m["reward"].astype(np.float64))           # (2000, 125)
        # probe
        rng = np.random.default_rng(0)
        peps = np.sort(rng.choice(N_TRAIN, a.probe_eps, replace=False))
        X = lat[peps].reshape(-1, D); Y = block_targets(td, peps).reshape(len(X), -1)
        G = np.repeat(np.arange(len(peps)), N_LAT)
        folds = list(GroupKFold(5).split(X, Y, G))
        ridge = RidgeCV(alphas=ALPHAS, cv=folds).fit(X, Y)
        eeps = np.arange(N_TRAIN, 2000)
        Xe = lat[eeps].reshape(-1, D); Ye = block_targets(td, eeps).reshape(len(Xe), -1)
        P, Q = td.n_prop, td.n_prop + td.n_pos
        ph = ridge.predict(Xe)
        probe_r2 = {"proprio": group_r2(ph[:, :P], Ye[:, :P]), "qpos": group_r2(ph[:, P:Q], Ye[:, P:Q]),
                    "qvel": group_r2(ph[:, Q:], Ye[:, Q:]), "state": group_r2(ph[:, P:], Ye[:, P:])}
        # reward head
        torch.manual_seed(0)
        head = RewardHead(D).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=3e-4)
        Xr = torch.as_tensor(lat[:N_TRAIN, 1:].reshape(-1, D), device=dev)
        Yr = symlog(torch.as_tensor(rew[:N_TRAIN].reshape(-1), dtype=torch.float32, device=dev))
        g = torch.Generator(device=dev).manual_seed(0)
        for s in range(a.steps):
            idx = torch.randint(0, len(Xr), (1024,), device=dev, generator=g)
            loss = ((head(Xr[idx]) - Yr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            Xv = torch.as_tensor(lat[N_TRAIN:, 1:].reshape(-1, D), device=dev)
            pr = symexp(torch.cat([head(Xv[i:i + 8192]) for i in range(0, len(Xv), 8192)])).cpu().numpy()
        yv = rew[N_TRAIN:].reshape(-1)
        reward_r2 = float(1 - ((pr - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
        # kinematic constant over one block, and D_eval latent variance
        q, v = td.qpos[:N_TRAIN], td.qvel[:N_TRAIN]
        dp = (q[:, BLOCK:] - q[:, :-BLOCK]).reshape(-1, q.shape[-1]); vv = v[:, :-BLOCK].reshape(-1, v.shape[-1])
        kin4 = (vv * dp).sum(0) / (vv * vv).sum(0)
        var = float(lat[N_TRAIN:].reshape(-1, D).var(0).sum())
        torch.save({"task": task, "probe_coef": ridge.coef_, "probe_intercept": ridge.intercept_,
                    "probe_alpha": float(ridge.alpha_), "n_prop": td.n_prop, "probe_r2_heldout": probe_r2,
                    "reward_state_dict": {k: x.cpu() for k, x in head.state_dict().items()},
                    "reward_r2_heldout": reward_r2, "kin4": kin4, "latent_var_eval": var,
                    "recipe": {"probe_eps": a.probe_eps, "reward_steps": a.steps}}, OUT / f"{task}.pt")
        print(f"{task}: probe alpha {ridge.alpha_:g}, held-out R2 {json.dumps({k: round(x, 3) for k, x in probe_r2.items()})}; "
              f"reward head held-out R2 {reward_r2:.3f}; latent var {var:.1f}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
