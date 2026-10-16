"""Train one main-grid world model on a DMControl task (plan Sections 2.2, 4).

Checkpoints follow the plan's within-run schedule: log-spaced snapshots at
3, 6, 12, 25, 50 and 100 % of the step budget (Pop-W), plus every 10 % for
extra resolution.  Pop-B runs vary seed, data fraction (10 %, 30 %) and model
width (0.5x, 2x).  Checkpoints go to scratch (disposable once evaluated); the
run reports progress to results/progress/dmc/ on storage, which the live
dashboard shows.  Mixed precision uses fp16 with loss scaling: the V100
emulates bf16 at a tenth of fp16 speed (plan amendment A3).
"""
import sys, pathlib, json, time, argparse, datetime, traceback
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from wm.pixel import PixelRSSM, PixelJEPA, PixelJEPAHist, pixel_rssm_loss, pixel_jepa_loss, pixel_jepa_hist_loss
from wm.protocol import check_precision

# The environment comes from MT_ENV (default dmc), read once at import, so every module that
# imports these constants sees the same environment.  Meta-World: experiments/metaworld/.
import os
ENV = os.environ.get("MT_ENV") or "dmc"
_ENVS = {"dmc": dict(data="dmc64", ckpt="dmc", prog="dmc", n_ep=2000, n_train=1900, label="DMControl main grid"),
         "metaworld": dict(data="metaworld64", ckpt="metaworld", prog="metaworld", n_ep=10000, n_train=9500,
                           label="Meta-World main grid")}
_E = _ENVS[ENV]
_SCR = pathlib.Path(os.environ.get("RCH_SCRATCH", "/scratch/hpc/11/xiar3/MetricTrust"))   # side paper 16B
DATA = _SCR / "data" / _E["data"]
CKPT = _SCR / "checkpoints" / _E["ckpt"]
PROG = ROOT / "results" / "progress" / _E["prog"]
N_EPISODES, N_TRAIN = _E["n_ep"], _E["n_train"]      # the last 5 % of episodes are D_eval
LOG_FRACS = (0.03, 0.06, 0.12, 0.25, 0.5, 1.0)


class Progress:
    def __init__(self, family, task, run, seed, total, planned_eval):
        PROG.mkdir(parents=True, exist_ok=True)
        self.f = PROG / f"{family}_{task}_{run}.json"
        self.d = {"experiment": _E["label"], "family": family, "task": task, "run": run,
                  "seed": seed, "unit": "step", "done": 0, "total": total, "status": "running",
                  "evaluated": 0, "planned_eval": planned_eval}

    def write(self, **kw):
        self.d.update(kw, updated_iso=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        tmp = self.f.with_suffix(".tmp"); tmp.write_text(json.dumps(self.d)); tmp.replace(self.f)


_FRAMES = {}


def frames_in_ram(task):
    """The task's 64x64 frames, read into RAM once per process (about 12 GB).
    Random windows through a memory map were the training bottleneck."""
    if task not in _FRAMES:
        _FRAMES[task] = np.load(DATA / task / "frames.npy")
    return _FRAMES[task]


class Windows:
    def __init__(self, task, episodes, T, rng, prefetch=False, B=16, dev="cuda"):
        self.frames = frames_in_ram(task)
        m = np.load(DATA / task / "meta.npz")
        self.act, self.rew = m["action"].astype(np.float32), m["reward"].astype(np.float32)
        self.eps, self.T, self.rng = np.asarray(episodes), T, rng
        self.L = self.act.shape[1]
        self.q = None
        if prefetch:                                   # overlap host gathering with GPU compute
            import queue, threading
            self.q = queue.Queue(maxsize=3)
            def work():
                while True:
                    self.q.put(self._gather(B, dev))
            threading.Thread(target=work, daemon=True).start()

    def _gather(self, B, dev):
        e = self.rng.choice(self.eps, B)
        s = self.rng.integers(0, self.L - self.T + 1, B)
        ti = s[:, None] + np.arange(self.T + 1)[None]             # one vectorised gather per array
        o = self.frames[e[:, None], ti]
        a = self.act[e[:, None], ti[:, :-1]]; r = self.rew[e[:, None], ti[:, :-1]]
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).pin_memory().to(dev, non_blocking=True)
        return t(o), t(a), t(r), torch.zeros(B, self.T, dtype=torch.bool, device=dev)

    def batch(self, B, dev):
        return self.q.get() if self.q is not None else self._gather(B, dev)


def schedule(steps):
    fr = set(LOG_FRACS) | {round(0.1 * k, 1) for k in range(1, 10)}
    return sorted({0} | {max(1, int(round(f * steps))) for f in fr})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--family", required=True, choices=["rssm", "jepa"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--data-frac", type=float, default=1.0); ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--B", type=int, default=16); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--tag", default="")
    ap.add_argument("--arch", default="single", choices=["single", "hist3"],
                    help="jepa only: hist3 = 3-frame history predictor (plan amendment A11)")
    ap.add_argument("--val-every", type=int, default=0, help="extra validation points (budget pilots)")
    a = ap.parse_args()
    run = f"seed{a.seed}" + (f"_d{int(a.data_frac * 100)}" if a.data_frac < 1 else "") + \
          (f"_w{a.width:g}" if a.width != 1 else "") + a.tag
    is_popw = a.data_frac == 1 and a.width == 1
    prog = Progress(a.family, a.task, run, a.seed, a.steps, len(LOG_FRACS) if is_popw else 1)
    try:
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        dev = "cuda"
        amp_dtype = torch.float16 if check_precision() == "fp16" else torch.bfloat16
        check_precision(autocast_dtype=amp_dtype)
        train_eps = np.arange(N_TRAIN)
        if a.data_frac < 1:
            train_eps = np.sort(rng.permutation(N_TRAIN)[:int(a.data_frac * N_TRAIN)])
        ev = Windows(a.task, np.arange(N_TRAIN, N_EPISODES), a.T, np.random.default_rng(999))
        tr = Windows(a.task, train_eps, a.T, rng, prefetch=True, B=a.B, dev=dev)
        act_dim = tr.act.shape[-1]
        Model, loss_fn = {"rssm": (PixelRSSM, pixel_rssm_loss), "jepa": (PixelJEPA, pixel_jepa_loss),
                          "jepa_hist3": (PixelJEPAHist, pixel_jepa_hist_loss)}[
            a.family + ("_hist3" if a.family == "jepa" and a.arch == "hist3" else "")]
        model = Model(act_dim=act_dim, width=a.width).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
        out = CKPT / a.family / a.task / run; out.mkdir(parents=True, exist_ok=True)
        ev_batches = [ev.batch(a.B, dev) for _ in range(6)]
        save_at = set(schedule(a.steps)); log = []; t0 = time.time()
        print(f"{a.family} {a.task} {run}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, "
              f"{len(train_eps)} train episodes, {a.steps} steps, amp {amp_dtype}, saves at {sorted(save_at)}", flush=True)
        prog.write()

        def val_loss():
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                vl = float(np.mean([loss_fn(model, *b)[1]["loss"] for b in ev_batches]))
            model.train()
            return vl

        def save(step):
            vl = val_loss()
            torch.save({"state_dict": model.state_dict(), "step": step, "frac": step / a.steps, "family": a.family,
                        "task": a.task, "run": run, "seed": a.seed, "width": a.width, "data_frac": a.data_frac,
                        "act_dim": act_dim, "val_loss": vl, "arch": a.arch}, out / f"step{step:07d}.pt")
            log.append({"step": step, "val_loss": vl, "t": time.time() - t0})
            print(f"  step {step:6d}  val_loss {vl:10.3f}  {time.time()-t0:6.0f}s", flush=True)

        save(0)
        for step in range(1, a.steps + 1):
            with torch.autocast("cuda", dtype=amp_dtype):
                loss, info = loss_fn(model, *tr.batch(a.B, dev))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {info}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            scaler.step(opt); scaler.update()
            if step in save_at:
                save(step)
            elif a.val_every and step % a.val_every == 0:
                vl = val_loss()
                log.append({"step": step, "val_loss": vl, "t": time.time() - t0, "saved": False})
                print(f"  step {step:6d}  val_loss {vl:10.3f}  (not saved)", flush=True)
            if step % 500 == 0 or step in save_at:
                prog.write(done=step)
        json.dump(log, open(out / "train_log.json", "w"), indent=1)
        prog.write(done=a.steps, status="done")
    except Exception as e:
        prog.write(status="failed", error=f"{type(e).__name__}: {e}"[:300])
        traceback.print_exc(); raise


if __name__ == "__main__":
    main()
