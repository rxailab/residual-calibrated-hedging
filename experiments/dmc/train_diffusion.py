"""Fine-tune MiniWorld-0.5B (the public DROID checkpoint) on DMControl: the
video-diffusion family of the main grid (plan Section 4).

One model per environment, covering its three tasks with task conditioning
(plan 2.2), trained on the same D_train episodes as every other family, read as
Wan2.2-VAE latents pre-encoded at 128x128 by scripts/diffusion_encode.py.
Snapshots at 3, 6, 12, 25, 50 and 100 % of the step budget plus every 10 %;
Pop-B runs vary seed and data fraction (10 %, 30 %, the same per-task episode
subsets as the other families).

The objective, model and noise schedule are MiniWorld's (third_party/MiniWorld
at e484206: miniworld/denoiser.py, miniworld/miniworld.py, miniworld/train.py):
block-causal diffusion forcing with flow matching on chunks of 2 latent frames,
logit-normal timesteps with the token-count shift, 10 % condition dropout for
classifier-free guidance, Muon with an auxiliary AdamW, gradient clipping at 1,
bf16 autocast, an EMA copy used for sampling.

Choices and deviations, for the pre-registration log:
 1. Actions.  MiniWorld conditions latent frame k >= 1 on the 4 raw actions
    that produce its 4 frames, 7 numbers each for DROID (28 in all).  Each
    DMControl decision's action (1, 6 or 2 numbers) is zero-padded into the 7
    slots, and a 3-way task one-hot is appended (cond_dim 31).  The new input
    columns of the action encoder and of the learned null action start at
    zero, so the network at step 0 computes exactly the pretrained function.
 2. Clips of 16 latent frames (61 frames, 60 decisions) cut from whole
    episodes encoded as one causal video; the first latent frame of a clip
    takes the learned null action, as every clip's first frame does upstream.
 3. Learning rate 1e-4, upstream's rate for training on a new dataset (its
    continued-training stages use 2e-5); the DMControl action space is new to
    the model.  EMA decay 0.999 instead of 0.9999: these runs are tens of
    thousands of steps, not hundreds of thousands, and with 0.9999 the early
    snapshots would be mostly the pretrained weights.
 4. No colour augmentation (rendered frames have fixed colours).
 5. Validation loss: the flow-matching loss on 48 fixed D_eval clips (16 per
    task) with fixed noise, the first chunk clean as context and every later
    chunk at each of t = 0.25, 0.5, 0.75 and 0.95 in turn: deterministic and
    comparable across checkpoints.
 6. Grid runs start before the budget pilot ends (amendment A17).  With
    --budgets lo:hi:step a run snapshots at every step that any of those
    candidate budgets would snapshot (the log-spaced fractions for Pop-W runs,
    the final step for Pop-B runs), records no fraction, and keeps its state at
    the end so that it can be extended.  The learning rate is constant and
    snapshots draw no random numbers, so the run is the one --steps B would
    give; scripts/diffusion_select.py keeps the snapshots of the chosen B.
"""
import os, sys, pathlib, json, time, argparse, signal, subprocess, traceback, copy
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "third_party" / "MiniWorld", ROOT / "third_party" / "shims"):
    sys.path.insert(0, str(p))
import numpy as np, torch
from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
from miniworld.muon import MuonWithAuxAdam, get_muon_param_groups
from experiments.dmc.train import N_TRAIN, LOG_FRACS, Progress, schedule

LAT = pathlib.Path("/scratch/hpc/11/xiar3/MetricTrust/data/dmc_lat")
META = pathlib.Path("/scratch/hpc/11/xiar3/MetricTrust/data/dmc64")
CKPT = pathlib.Path("/scratch/hpc/11/xiar3/MetricTrust/checkpoints/dmc/diffusion/dmcontrol")
PRETRAINED = "/scratch/hpc/11/xiar3/MetricTrust/pretrained/MiniWorld_0_5b_droid.pt"
TASKS = ("cartpole-swingup", "walker-walk", "cup-catch")
A_SLOT = 7                                   # DROID's per-frame action width
COND_DIM = 4 * A_SLOT + len(TASKS)           # 4 decisions per latent frame + task one-hot
N_LAT = 126                                  # latent frames per 501-frame episode
VAL_T = (0.25, 0.5, 0.75, 0.95)


def make_config(L=16, **kw):
    cfg = dict(wm_model="0.5B", latent_size=(8, 8), latent_channels=48, latent_frames=L,
               cond_dim=COND_DIM, wm_use_checkpoint=False, cond_dropout_prob=0.1,
               adaln_mode="adaln_lora", df_chunk_size=2, df_ardiff_step=10,
               timestep_baseshift=2.667, cfg_scale=1.0)
    cfg.update(kw)
    return DenoiserConfig(**cfg)


def load_pretrained(den, path=PRETRAINED):
    """The DROID weights, with the action encoder's input and the null action
    widened from 28 to COND_DIM by zero columns (design note 1).  RoPE tables
    are rebuilt for this resolution and clip length, not loaded."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("net.action_encoder.to_emb.0.weight", "net.action_encoder.to_mod.0.weight"):
        w = sd[k]
        sd[k] = torch.cat([w, torch.zeros(w.shape[0], COND_DIM - w.shape[1], dtype=w.dtype)], 1)
    sd["net.null_action"] = torch.cat([sd["net.null_action"],
                                       torch.zeros(1, 1, COND_DIM - sd["net.null_action"].shape[-1])], -1)
    own = den.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(sd) - set(keep)); missing = sorted(set(own) - set(keep))
    allowed = {"net.feat_rope.freqs_cos", "net.feat_rope.freqs_sin"}
    if set(skipped) - allowed or set(missing) - allowed:
        raise RuntimeError(f"pretrained load mismatch: skipped {skipped}, missing {missing}")
    den.load_state_dict(keep, strict=False)
    return {"loaded": len(keep), "rebuilt": missing}


def cond_table(actions, task_idx):
    """(E, 500, A) actions -> (E, 126, COND_DIM) per-latent-frame conditions:
    latent k >= 1 gets decisions 4k-4 .. 4k-1, zero-padded to 7 slots each;
    latent 0 (the episode's seed frame) gets zeros; every frame the task one-hot."""
    E, T, A = actions.shape
    pad = np.zeros((E, T, A_SLOT), np.float32); pad[..., :A] = actions
    out = np.zeros((E, N_LAT, COND_DIM), np.float32)
    out[:, 1:, :4 * A_SLOT] = pad.reshape(E, T // 4, 4 * A_SLOT)
    out[..., 4 * A_SLOT + task_idx] = 1.0
    return out


class LatentData:
    """All three tasks' latents (float16, about 4.6 GB) and conditions in RAM."""

    def __init__(self, tasks=TASKS):
        self.lat, self.cond = [], []
        for i, task in enumerate(tasks):
            lat = np.concatenate([np.load(LAT / task / f"lat_{k:03d}.npy") for k in range(40)])
            assert lat.shape == (2000, N_LAT, 48, 8, 8), lat.shape
            self.lat.append(lat)
            self.cond.append(cond_table(np.load(META / task / "meta.npz")["action"].astype(np.float32), i))

    def clips(self, t, e, k0, L):
        lat = np.stack([self.lat[ti][ei, k:k + L] for ti, ei, k in zip(t, e, k0)])   # (B, L, 48, 8, 8)
        cond = np.stack([self.cond[ti][ei, k:k + L] for ti, ei, k in zip(t, e, k0)])
        return lat, cond


def to_dev(lat, cond, dev):
    lat = torch.from_numpy(np.ascontiguousarray(lat)).to(dev, non_blocking=True).float().permute(0, 2, 1, 3, 4)
    return lat.contiguous(), torch.from_numpy(cond).to(dev, non_blocking=True)


def val_set(data, L, n_per_task=16, seed=4242):
    rng = np.random.default_rng(seed)
    t = np.repeat(np.arange(len(TASKS)), n_per_task)
    e = rng.integers(N_TRAIN, 2000, len(t)); k0 = rng.integers(0, N_LAT - L + 1, len(t))
    lat, cond = data.clips(t, e, k0, L)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn((len(t), 48, L, 8, 8), generator=g)
    return lat, cond, noise


@torch.no_grad()
def val_loss(den, vs, dev, amp, chunk=2, bs=16):
    """Design note 5."""
    lat_np, cond_np, noise_all = vs
    net = den.net; net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(lat_np), bs):
        x, c = to_dev(lat_np[i:i + bs], cond_np[i:i + bs], dev)
        eps = noise_all[i:i + bs].to(dev)
        b, _, L = x.shape[:3]
        for tv in VAL_T:
            tf = torch.full((b, L), tv, device=dev); tf[:, :chunk] = 0.0
            tw = tf.view(b, 1, L, 1, 1)
            z = (1 - tw) * x + tw * eps
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                v = net(z, tf, c, temporal_causal=True, chunk_size=chunk,
                        cond_drop=torch.zeros(b, dtype=torch.bool, device=dev))
            d = ((x - eps) - v.float()) ** 2
            tot += float(d[:, :, chunk:].mean(dim=(1, 3, 4)).mean(1).sum()); n += b
    return tot / n


def candidate_steps(budgets, popw=True):
    """Design note 6: every step a run would snapshot under any candidate budget."""
    fr = LOG_FRACS if popw else (1.0,)
    return sorted({max(1, int(round(f * b))) for b in budgets for f in fr})


def parse_budgets(s):
    lo, hi, st = map(int, s.split(":"))
    return list(range(lo, hi + 1, st))


@torch.no_grad()
def update_ema(model, ema, decay):
    for p, q in zip(model.parameters(), ema.parameters()):
        q.lerp_(p.detach(), 1 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--data-frac", type=float, default=1.0)
    ap.add_argument("--B", type=int, default=32); ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--max-hours", type=float, default=0, help="save state and stop (requeue) after this")
    ap.add_argument("--budgets", default="", help="lo:hi:step candidate budgets (design note 6)")
    a = ap.parse_args()
    budgets = parse_budgets(a.budgets) if a.budgets else None
    run = f"seed{a.seed}" + (f"_d{int(a.data_frac * 100)}" if a.data_frac < 1 else "") + a.tag
    is_popw = a.data_frac == 1
    prog = Progress("diffusion", "dmcontrol", run, a.seed, a.steps, len(LOG_FRACS) if is_popw else 1)
    out = CKPT / run; out.mkdir(parents=True, exist_ok=True)
    state_f = out / "state.pt"
    if (out / "train_log.json").exists() and not state_f.exists():
        print(f"{run} already finished; nothing to do", flush=True)      # a duplicate submission
        return
    try:
        dev = "cuda"
        cap = torch.cuda.get_device_capability(0)
        amp = torch.bfloat16 if cap >= (8, 0) else torch.float16
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        train_eps = np.arange(N_TRAIN)
        if a.data_frac < 1:                  # the same subset as train.py draws for this seed
            train_eps = np.sort(rng.permutation(N_TRAIN)[:int(a.data_frac * N_TRAIN)])
        data = LatentData()
        vs = val_set(data, a.L)
        # Activation checkpointing on GPUs under 80 GB (L40, L40S): same arithmetic, recomputed.
        ckpt_act = torch.cuda.get_device_properties(0).total_memory < 80e9
        den = build_denoiser_from_mode(make_config(a.L, wm_use_checkpoint=ckpt_act)).to(dev)
        info = load_pretrained(den)
        ema = copy.deepcopy(den).requires_grad_(False).eval()
        opt = MuonWithAuxAdam(get_muon_param_groups(den, weight_decay=0.0, lr=a.lr))
        scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
        step, log = 0, {"val": [], "train": []}
        if state_f.exists():                 # resume after a requeue
            st = torch.load(state_f, map_location="cpu", weights_only=False)
            den.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
            scaler.load_state_dict(st["scaler"]); step, log = st["step"], st["log"]
            rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
            torch.cuda.set_rng_state(st["cuda_rng"])
            print(f"resumed at step {step}", flush=True)
        save_at = set(candidate_steps(budgets, is_popw) if budgets else schedule(a.steps))
        n_par = sum(p.numel() for p in den.parameters())
        print(f"diffusion {run}: MiniWorld-0.5B {n_par/1e9:.3f}B params, pretrained {info}, "
              f"{len(train_eps)} train episodes x {len(TASKS)} tasks, {a.steps} steps, batch {a.B} x {a.L} "
              f"latent frames, lr {a.lr}, ema {a.ema}, amp {amp}, activation checkpointing {ckpt_act}, "
              f"{torch.cuda.get_device_name(0)}", flush=True)
        prog.write(done=step)
        stop = {"flag": False}
        signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))
        t_start = time.time()

        def save_state():
            tmp = state_f.with_suffix(".tmp")
            torch.save({"model": den.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "log": log, "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, tmp)
            tmp.replace(state_f)

        def snapshot(s):
            v = val_loss(ema, vs, dev, amp)
            torch.save({"state_dict": {k: x.to(torch.bfloat16) if x.is_floating_point() else x
                                       for k, x in ema.state_dict().items()},
                        "config": vars(make_config(a.L)), "step": s, "family": "diffusion",
                        "frac": float("nan") if budgets else s / a.steps, "budget": None if budgets else a.steps,
                        "task": "dmcontrol", "run": run, "seed": a.seed, "data_frac": a.data_frac, "width": float("nan"),
                        "val_loss": v, "lr": a.lr, "ema_decay": a.ema, "tasks": TASKS, "a_slot": A_SLOT},
                       out / f"step{s:07d}.pt")
            log["val"].append({"step": s, "val_loss": v, "saved": True, "t": time.time() - t_start})
            print(f"  step {s:7d}  val {v:.5f}  {time.time()-t_start:6.0f}s", flush=True)

        if step == 0 and not (out / "step0000000.pt").exists():
            snapshot(0)
        acc, t0, last_state = [], time.time(), time.time()
        while step < a.steps:
            t = rng.integers(0, len(TASKS), a.B); e = rng.choice(train_eps, a.B)
            k0 = rng.integers(0, N_LAT - a.L + 1, a.B)
            x, c = to_dev(*data.clips(t, e, k0, a.L), dev)
            den.train()
            with torch.autocast("cuda", dtype=amp):
                loss = den(x, c)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            update_ema(den, ema, a.ema)
            step += 1; acc.append((float(loss), float(gn)))
            if step in save_at:
                snapshot(step)
            elif a.val_every and step % a.val_every == 0:
                v = val_loss(ema, vs, dev, amp)
                log["val"].append({"step": step, "val_loss": v, "saved": False, "t": time.time() - t_start})
                print(f"  step {step:7d}  val {v:.5f}", flush=True)
            if step % 100 == 0:
                l, g = np.mean(acc, 0); acc = []
                log["train"].append({"step": step, "loss": float(l), "grad_norm": float(g)})
                if step % 500 == 0:
                    print(f"  train {step:7d}  loss {l:.5f} |g| {g:.3f}  {(time.time()-t0)/100:.2f} s/step "
                          f"(last 100)", flush=True)
                t0 = time.time()
                prog.write(done=step)
            if time.time() - last_state > 1800 and step < a.steps:          # plan Section 12: state every 30 min
                save_state(); last_state = time.time()
            over = a.max_hours and time.time() - t_start > a.max_hours * 3600
            if (stop["flag"] or over) and step < a.steps:
                save_state(); prog.write(done=step, status="queued")
                print(f"stopping at step {step} for requeue", flush=True)
                if os.environ.get("SLURM_JOB_ID"):
                    jid = os.environ["SLURM_JOB_ID"]      # for an array task its own name: a bare id that
                    if os.environ.get("SLURM_ARRAY_TASK_ID"):   # equals the array's would requeue every task
                        jid = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
                    subprocess.run(["scontrol", "requeue", jid], check=False)
                return
        json.dump(log, open(out / "train_log.json", "w"), indent=1)
        if state_f.exists() and not budgets:          # kept under --budgets: the run may be extended
            state_f.unlink()
        prog.write(done=a.steps, status="done")
    except Exception as ex:
        prog.write(status="failed", error=f"{type(ex).__name__}: {ex}"[:300])
        traceback.print_exc(); raise


if __name__ == "__main__":
    main()
