"""Study 3 (diffusion part): delay-in-decisions closed loop for the MiniWorld
diffusion world model, and its single-environment call latency.

The model advances in blocks of 4 decisions, so plans start at block
boundaries and every delay and replanning interval is a multiple of 4.  The
closed loop is evaluate_diffusion.u1_planning's (streaming VAE encoder,
context of 4 latent frames, task-conditioned blocks, shared reward head, D11
handling of non-finite costs) with the planning schedule of latency/schedule.py.

  closed loop:  run_diffusion.py CKPT --task T --arms paused8,cpp8 --offset 0 --episodes 5
  timing:       run_diffusion.py CKPT --task T --timing

Results: results/latency/diffusion/<task>/<run>/s<steps>/<arm>_e<offset>.json and
results/latency/timing/<gpu>/diffusion_<task>_<run>.json.
"""
import argparse
import dataclasses
import datetime
import json
import os
import re
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import wm.diffusion as WD  # noqa: E402
from experiments.dmc import evaluate_diffusion as EVD  # noqa: E402  (imported, never modified)
from envs.dmc import DMCEnv  # noqa: E402
from latency.run_dmc import SEED0, git_hash  # noqa: E402
from latency.schedule import MPPIConfig, PrefixMPPI, Timer, parse_arm, run_schedule  # noqa: E402
from latency.timing import gpu_tag  # noqa: E402
from paths import RESULTS  # noqa: E402

BLOCK = WD.BLOCK


class DiffusionBatch:
    def __init__(self, vae, dt, seeds, S, dev, render=True):
        self.E, self.dt, self.S = len(seeds), dt, S
        self.envs = [DMCEnv(dt.task, img=128, render=render) for _ in seeds]
        frames0 = np.stack([e.reset(s)["pixels"] for e, s in zip(self.envs, seeds)])
        self.enc = WD.StreamEncoder(vae, dev)
        self.lat_hist = [self.enc.reset(frames0).cpu().numpy()]
        self.cond_hist = [WD.seed_cond(self.E, dt.ti).numpy()[:, 0]]
        self.ret = np.zeros(self.E)
        self.fr, self.acts = [], []
        self.t_env = 0.0

    def latent(self):
        assert not self.fr, "plans start at block boundaries"
        C_ = min(self.S.ctx, len(self.lat_hist))
        return np.stack(self.lat_hist[-C_:], 1), np.stack(self.cond_hist[-C_:], 1)

    def step(self, act):
        a = act.float().cpu().numpy()
        t0 = time.perf_counter()
        out = [e.step(x) for e, x in zip(self.envs, a)]
        self.ret += np.array([r for _, r, _ in out])
        self.fr.append(np.stack([o["pixels"] for o, _, _ in out])); self.acts.append(a)
        if len(self.fr) == BLOCK:
            self.lat_hist.append(self.enc.push(np.stack(self.fr, 1)).cpu().numpy())
            self.cond_hist.append(WD.cond_blocks(np.stack(self.acts, 1)[:, None], self.dt.ti).numpy()[:, 0])
            self.fr, self.acts = [], []
        self.t_env += time.perf_counter() - t0


def make_cost_factory(samp, dt, P):
    def make_cost(Z):
        ctx, cc = Z
        ctx_r, cc_r = np.repeat(ctx, P, 0), torch.as_tensor(np.repeat(cc, P, 0))

        def cost(seq):
            n, L, A = seq.shape
            assert L % BLOCK == 0
            fc = WD.cond_blocks(seq.reshape(n, L // BLOCK, BLOCK, A).float(), dt.ti)
            pred = samp(ctx_r, cc_r, fc, None)
            return -dt.block_reward(pred).sum(1).double()
        return cost
    return make_cost


def residual_bank_diffusion(samp, dt, S, d, n=128, seed=11):
    """(n, d/4, 48, 8, 8) residuals of the sampler's d-step prediction through recorded
    actions, true latent - predicted latent, on D_train windows (episodes 0-99, the first
    two pre-encoded shards).  Same role as latency/hedge.residual_bank."""
    from experiments.dmc.train_diffusion import LAT, cond_table
    db = d // BLOCK
    lat = np.concatenate([np.load(LAT / dt.task / f"lat_{k:03d}.npy") for k in (0, 1)]).astype(np.float32)
    act = np.load(EVD.EV.DATA / dt.task / "meta.npz")["action"][:len(lat)].astype(np.float32)
    cond = cond_table(act, dt.ti)
    rng = np.random.default_rng(seed + d)
    e = rng.integers(0, len(lat), n)
    b = rng.integers(S.ctx - 1, lat.shape[1] - db - 1, n)
    W = S.ctx
    ctx = np.stack([lat[i, j - W + 1:j + 1] for i, j in zip(e, b)])
    cc = torch.as_tensor(np.stack([cond[i, j - W + 1:j + 1] for i, j in zip(e, b)]))
    fc = torch.as_tensor(np.stack([cond[i, j + 1:j + 1 + db] for i, j in zip(e, b)]))
    true = np.stack([lat[i, j + 1:j + 1 + db] for i, j in zip(e, b)])
    pred = samp(ctx, cc, fc, None).cpu().numpy()
    return true - pred


def make_cost_hedged_diffusion(samp, dt, S, P, d, bank, K, H, gen_seed=5):
    """Residual-hedged committed-prefix scoring for the diffusion model.  The committed
    prefix (d/4 blocks) is sampled once per episode; each anchor is the context window
    (last S.ctx latent frames) with the predicted prefix frames plus one bank residual
    (none for the first anchor); every candidate's free part is scored from each anchor
    and the predicted costs are averaged.  The pre-episode plan has no prefix."""
    rng = np.random.default_rng(gen_seed)
    db = d // BLOCK
    bank_t = torch.as_tensor(bank, dtype=torch.float32)

    def make_cost(Z):
        ctx, cc = Z
        E = len(ctx)

        def cost(seq):
            n, L, A = seq.shape
            if L == H:                                              # pre-episode plan
                fc = WD.cond_blocks(seq.reshape(n, L // BLOCK, BLOCK, A).float(), dt.ti)
                pred = samp(np.repeat(ctx, P, 0), torch.as_tensor(np.repeat(cc, P, 0)), fc, None)
                return -dt.block_reward(pred).sum(1).double()
            assert L == H + d
            s = seq.reshape(E, P, L, A)
            fcp = WD.cond_blocks(s[:, 0, :d].reshape(E, db, BLOCK, A).float(), dt.ti)          # (E, db, C)
            pp = samp(ctx, torch.as_tensor(cc), fcp, None).cpu()                                 # (E, db, ...)
            cc_new = torch.cat([torch.as_tensor(cc), fcp.cpu()], 1)[:, -S.ctx:]
            fc_free = WD.cond_blocks(s[:, :, d:].reshape(E * P, H // BLOCK, BLOCK, A).float(), dt.ti)
            idx = rng.integers(0, len(bank), (E, max(K - 1, 1)))
            J = 0.0
            for k in range(K):
                pk = pp if k == 0 else pp + bank_t[idx[:, k - 1]]
                ck = torch.cat([torch.as_tensor(ctx, dtype=torch.float32), pk], 1)[:, -S.ctx:].numpy()
                pred = samp(np.repeat(ck, P, 0), cc_new.repeat_interleave(P, 0), fc_free, None)
                J = J + (-dt.block_reward(pred).sum(1).double())
            return J / K
        return cost
    return make_cost


def load(ckpt, task, steps, dev, overrides=()):
    S = EVD.Settings(steps=steps)
    for kv in overrides:                       # smoke tests only, e.g. pop=4 iters=1 h_plan=2
        k, v = kv.split("=")
        setattr(S, k, type(getattr(S, k))(v))
    den, ck = WD.load_denoiser(ckpt, dev)
    vae = WD.load_vae(dev)
    dt = EVD.DiffTask(task, dev)
    samp = WD.Sampler(den, steps=S.steps, ardiff=S.ardiff, cfg_scale=S.cfg_scale, micro_batch=S.micro_batch)
    meta = {k: ck.get(k) for k in ("family", "run", "seed", "data_frac", "step", "val_loss")}
    meta["task"] = task
    return S, vae, dt, samp, meta


def timing(ckpt, task, dev, calls=5, warmup=2):
    out = {"ckpt": ckpt, "gpu": torch.cuda.get_device_name(0), "calls": calls, "warmup": warmup,
           "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"), "git": git_hash(), "results": []}
    for steps in (2, 4, 8):
        S, vae, dt, samp, meta = load(ckpt, task, steps, dev)
        out["meta"] = meta
        env = DiffusionBatch(vae, dt, [SEED0], S, dev)
        Z = env.latent()
        cfg = MPPIConfig(population=S.pop, iterations=S.iters)
        planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(0), dev)
        H, A = BLOCK * S.h_plan, dt.td.act_dim
        for d in (0, 8, 16):
            warm = torch.zeros((1, H, A), dtype=torch.float64, device=dev)
            prefix = torch.zeros((1, d, A), dtype=torch.float64, device=dev) if d else None
            ts = []
            for k in range(warmup + calls):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                mu = planner(make_cost_factory(samp, dt, S.pop)(Z), warm, prefix)
                mu[:, :max(d, BLOCK)].float().cpu().numpy()
                torch.cuda.synchronize()
                if k >= warmup:
                    ts.append(time.perf_counter() - t0)
            ts = np.asarray(ts)
            out["results"].append({"steps": steps, "pop": S.pop, "iters": S.iters, "d": d,
                                   "L50_s": float(np.median(ts)), "L95_s": float(np.percentile(ts, 95)),
                                   "mean_s": float(ts.mean()), "samples_s": ts.tolist()})
            print(f"diffusion/{task} s{steps} d{d}: L50 {np.median(ts):.1f} s", flush=True)
        del samp, vae
        torch.cuda.empty_cache()
    out["written"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    od = RESULTS / "latency" / "timing" / gpu_tag(out["gpu"])
    od.mkdir(parents=True, exist_ok=True)
    f = od / f"diffusion_{task}_{out['meta']['run']}.json"
    f.write_text(json.dumps(out, indent=1)); print("wrote", f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--task", required=True)
    ap.add_argument("--arms", default="paused8,paused16,aligned8,cpp8,aligned16,cpp16")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0, help="first seed = 950,000 + offset")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override Settings (smoke tests)")
    ap.add_argument("--out", default=str(RESULTS / "latency" / "diffusion"))
    a = ap.parse_args()
    dev = "cuda"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    if a.timing:
        return timing(a.ckpt, a.task, dev)
    S, vae, dt, samp, meta = load(a.ckpt, a.task, a.steps, dev, a.set)
    cfg = MPPIConfig(population=S.pop, iterations=S.iters)
    H, A = BLOCK * S.h_plan, dt.td.act_dim
    od = pathlib.Path(a.out) / a.task / meta["run"] / f"s{a.steps}"
    od.mkdir(parents=True, exist_ok=True)
    seeds = list(range(SEED0 + a.offset, SEED0 + a.offset + a.episodes))
    failed = []
    for arm in a.arms.split(","):
        m_h = re.fullmatch(r"hcpp(\d+)k(\d+)", arm)
        sched = f"cpp{m_h.group(1)}" if m_h else arm
        assert parse_arm(sched)[1] % BLOCK == 0, "diffusion delays and intervals are whole blocks"
        f = od / f"{arm}_e{a.offset}.json"
        if f.exists() and not os.environ.get("FORCE"):
            print("skip (done)", f, flush=True); continue
        torch.manual_seed(777)                     # the sampler's own noise (no generator passed)
        env = DiffusionBatch(vae, dt, seeds, S, dev)
        planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(777), dev)
        timer = Timer(dev)
        t0 = time.time()
        try:
            if m_h:                                # residual-hedged committed prefix
                dh, K = int(m_h.group(1)), int(m_h.group(2))
                bank = residual_bank_diffusion(samp, dt, S, dh)
                mc = make_cost_hedged_diffusion(samp, dt, S, S.pop, dh, bank, K, H)
            else:
                mc = make_cost_factory(samp, dt, S.pop)
            n_calls = run_schedule(sched, H, a.T, planner, mc, env, A, timer)
        except Exception:
            traceback.print_exc(); failed.append(arm); continue
        pc = np.asarray(timer.per_call)
        r = {"arm": arm, "returns": env.ret.tolist(), "return_mean": float(env.ret.mean()),
             "episodes": a.episodes, "seeds": [seeds[0], seeds[-1]], "decisions": a.T, "H": H,
             "n_calls": n_calls, "t_plan_s": float(pc.sum()), "t_plan_call_mean_s": float(pc.mean()),
             "t_env_s": env.t_env, "t_wall_s": time.time() - t0,
             "nonfinite_frac": planner.n_bad / max(planner.n_all, 1), "batch_size": a.episodes,
             "settings": dataclasses.asdict(S), "config": dataclasses.asdict(cfg), "meta": meta, "ckpt": a.ckpt,
             "gpu": torch.cuda.get_device_name(0), "git": git_hash(), "host": os.uname().nodename,
             "job": os.environ.get("SLURM_JOB_ID"),
             "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        tmp = f.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(r, indent=1)); tmp.replace(f)
        print(f"diffusion/{a.task} {arm} e{a.offset}: return {r['return_mean']:.1f} ({n_calls} calls, "
              f"{r['t_plan_call_mean_s']:.0f} s/call, {r['t_wall_s'] / 3600:.2f} h)", flush=True)
    if failed:
        sys.exit(f"failed arms: {failed}")


if __name__ == "__main__":
    main()
