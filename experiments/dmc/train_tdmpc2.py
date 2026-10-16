"""Train one TD-MPC2 main-grid world model offline on a DMControl task
(plan Sections 2.2, 4; the model and update are wm/tdmpc2.py).

Same data, checkpoint schedule, progress reporting and storage policy as
train.py: D_train is episodes 0-1899 of the task's 64x64 frames, D_eval
1900-1999; snapshots at 3, 6, 12, 25, 50 and 100 % of the step budget plus
every 10 %.  Pop-B runs vary seed, data fraction (10 %, 30 %, the same episode
subsets as the other families) and model size (TD-MPC2's 1M and 19M).
Each update samples 256 windows of 4 consecutive decisions (horizon 3), each
observation a stack of 3 frames, as TD-MPC2's slice sampler does.
"""
import sys, pathlib, json, time, argparse, traceback
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from experiments.dmc.train import DATA, CKPT, N_TRAIN, N_EPISODES, LOG_FRACS, Progress, frames_in_ram, schedule
from wm.tdmpc2 import PixelTDMPC2, RunningScale, make_optimizers, update, val_losses, CFG


class StackWindows:
    """Windows of H+1 observations (3-frame stacks), H actions and H rewards.
    One contiguous gather of H+3 frames per window, stacked on the GPU."""

    def __init__(self, task, episodes, H, rng, prefetch=False, B=256, dev="cuda"):
        self.frames = frames_in_ram(task)
        m = np.load(DATA / task / "meta.npz")
        self.act, self.rew = m["action"].astype(np.float32), m["reward"].astype(np.float32)
        self.eps, self.H, self.rng, self.dev = np.asarray(episodes), H, rng, dev
        self.L = self.act.shape[1]
        self.nf = CFG["num_frames"]
        self.q = None
        if prefetch:
            import queue, threading
            self.q = queue.Queue(maxsize=4)
            def work():
                while True:
                    self.q.put(self._gather(B))
            threading.Thread(target=work, daemon=True).start()

    def _gather(self, B):
        e = self.rng.choice(self.eps, B)
        t = self.rng.integers(0, self.L - self.H + 1, B)
        nf, H = self.nf, self.H
        w = np.clip(t[:, None] - (nf - 1) + np.arange(H + nf)[None], 0, None)    # frames t-2 .. t+H
        fr = self.frames[e[:, None], w]                                          # (B, H+3, 64, 64, 3)
        ti = t[:, None] + np.arange(H)[None]
        a, r = self.act[e[:, None], ti], self.rew[e[:, None], ti]
        pin = lambda x: torch.from_numpy(np.ascontiguousarray(x)).pin_memory()
        return pin(fr), pin(a), pin(r)

    def batch(self, B):
        fr, a, r = self.q.get() if self.q is not None else self._gather(B)
        fr = fr.to(self.dev, non_blocking=True)
        a = a.to(self.dev, non_blocking=True).transpose(0, 1).contiguous()      # (H, B, A)
        r = r.to(self.dev, non_blocking=True).transpose(0, 1).unsqueeze(-1).contiguous()
        nf = self.nf
        st = torch.stack([fr[:, j:j + nf] for j in range(self.H + 1)])            # (H+1, B, nf, 64, 64, 3)
        st = st.permute(0, 1, 2, 5, 3, 4).reshape(self.H + 1, fr.shape[0], 3 * nf, 64, 64)
        return st, a, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--data-frac", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=5, choices=[1, 5, 19])
    ap.add_argument("--B", type=int, default=CFG["batch_size"]); ap.add_argument("--tag", default="")
    ap.add_argument("--val-every", type=int, default=0, help="extra validation points (pilot)")
    a = ap.parse_args()
    run = f"seed{a.seed}" + (f"_d{int(a.data_frac * 100)}" if a.data_frac < 1 else "") + \
          (f"_m{a.size}" if a.size != 5 else "") + a.tag
    is_popw = a.data_frac == 1 and a.size == 5
    prog = Progress("tdmpc2", a.task, run, a.seed, a.steps, len(LOG_FRACS) if is_popw else 1)
    try:
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        dev = "cuda"
        torch.backends.cudnn.benchmark = True
        train_eps = np.arange(N_TRAIN)
        if a.data_frac < 1:                  # the same subset as train.py draws for this seed
            train_eps = np.sort(rng.permutation(N_TRAIN)[:int(a.data_frac * N_TRAIN)])
        H = CFG["horizon"]
        ev = StackWindows(a.task, np.arange(N_TRAIN, N_EPISODES), H, np.random.default_rng(999), dev=dev)
        tr = StackWindows(a.task, train_eps, H, rng, prefetch=True, B=a.B, dev=dev)
        act_dim = tr.act.shape[-1]
        model = PixelTDMPC2(act_dim, size=a.size).to(dev)
        opt, pi_opt = make_optimizers(model)
        scale = RunningScale().to(dev)
        out = CKPT / "tdmpc2" / a.task / run; out.mkdir(parents=True, exist_ok=True)
        ev_batches = [ev.batch(a.B) for _ in range(6)]
        save_at = set(schedule(a.steps)); log, tlog = [], []; t0 = time.time()
        n_par = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("target_Qs"))
        print(f"tdmpc2 {a.task} {run}: {n_par/1e6:.2f}M params, {len(train_eps)} train episodes, "
              f"{a.steps} steps, batch {a.B}, fp32, saves at {sorted(save_at)}", flush=True)
        prog.write()

        def validate():
            vs = [val_losses(model, *b) for b in ev_batches]
            return {k: float(np.mean([v[k] for v in vs])) for k in vs[0]}

        def save(step):
            v = validate()
            torch.save({"state_dict": model.state_dict(), "scale": scale.state_dict(), "step": step,
                        "frac": step / a.steps, "family": "tdmpc2", "task": a.task, "run": run,
                        "seed": a.seed, "size": a.size, "width": float("nan"), "data_frac": a.data_frac,
                        "act_dim": act_dim, "val_loss": v["loss"], "val": v, "arch": f"tdmpc2_{a.size}M"},
                       out / f"step{step:07d}.pt")
            log.append({"step": step, "val_loss": v["loss"], "val": v, "t": time.time() - t0, "saved": True})
            print(f"  step {step:7d}  val {v['loss']:8.4f} (wm {v['wm']:.4f}, cons {v['consistency']:.5f}, "
                  f"rew {v['reward']:.4f}, value {v['value']:.4f})  {time.time()-t0:6.0f}s", flush=True)

        save(0)
        acc = []
        for step in range(1, a.steps + 1):
            info = update(model, opt, pi_opt, scale, *tr.batch(a.B))
            if not np.isfinite(info["loss"]):
                raise FloatingPointError(f"non-finite loss at step {step}: {info}")
            acc.append(info)
            if step in save_at:
                save(step)
            elif a.val_every and step % a.val_every == 0:
                v = validate()
                log.append({"step": step, "val_loss": v["loss"], "val": v, "t": time.time() - t0, "saved": False})
                print(f"  step {step:7d}  val {v['loss']:8.4f} (wm {v['wm']:.4f})  {time.time()-t0:6.0f}s", flush=True)
            if step % 1000 == 0:
                m = {k: float(np.mean([x[k] for x in acc])) for k in acc[0]}
                tlog.append({"step": step, **m}); acc = []
                if step % 5000 == 0:
                    print(f"  train {step:7d}  loss {m['loss']:.4f} cons {m['consistency']:.5f} "
                          f"rew {m['reward']:.4f} value {m['value']:.4f} pi {m['pi_loss']:.3f} "
                          f"scale {m['pi_scale']:.2f}  {(time.time()-t0)/step*1000:.1f} ms/step", flush=True)
            if step % 500 == 0 or step in save_at:
                prog.write(done=step)
        json.dump({"val": log, "train": tlog}, open(out / "train_log.json", "w"), indent=1)
        prog.write(done=a.steps, status="done")
    except Exception as e:
        prog.write(status="failed", error=f"{type(e).__name__}: {e}"[:300])
        traceback.print_exc(); raise


if __name__ == "__main__":
    main()
