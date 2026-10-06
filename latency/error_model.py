"""Delay-error model: feasibility test (side paper, method design).

A small heteroscedastic network is trained post hoc on a FROZEN world model's
own d-step prediction residuals,

    r = z(t+d) - rollout(z(t), a(t:t+d)),

conditioned on the predicted latent z_hat = rollout(...), the current latent
z(t) and the committed actions a(t:t+d).  It predicts the residual's mean and
per-dimension variance.  Training windows come from D_train only; the test
windows are the held-out D_eval windows of latency/diagnose_anchor.py.

Reported per delay d:
  err_left        |z - (z_hat + mu)|^2 / |z - z_hat|^2 on test windows
  regret_*        model-predicted regret from the true latent (as in
                  diagnose_anchor.py) of plans made from: the predicted anchor
                  (pred), the corrected anchor (corr), hedging over 4 anchors
                  from the unconditional residual bank around z_hat (uncond4,
                  the current method), and hedging over 4 anchors from the
                  learned distribution N(z_hat + mu, sigma^2) (cond4)
Writes results/latency/error_model/<family>_<task>_<run>.json.
"""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from experiments.dmc import evaluate as EV  # noqa: E402
from latency.diagnose_anchor import CTX, filtered, windows  # noqa: E402
from latency.run_dmc import H, load_adapter, make_cost_factory  # noqa: E402
from latency.schedule import MPPIConfig, PrefixMPPI  # noqa: E402
from paths import RESULTS  # noqa: E402


def train_windows(td, n, d_max, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, EV.N_TRAIN, n), rng.integers(CTX + 8, td.L - H - d_max - 2, n)


class ErrorNet(nn.Module):
    def __init__(self, n_in, D, hid=512):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU())
        self.mu, self.lv = nn.Linear(hid, D), nn.Linear(hid, D)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)

    def forward(self, x):
        h = self.f(x)
        return self.mu(h), self.lv(h).clamp(-12, 6)


def features(zh, zt, prefix):
    return np.concatenate([zh, zt, prefix.reshape(len(prefix), -1)], 1)


def fit(Xtr, Rtr, dev, steps=3000, bs=256, seed=0):
    torch.manual_seed(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-6
    rm, rs = Rtr.mean(0), Rtr.std(0) + 1e-6
    X = torch.as_tensor((Xtr - xm) / xs, dtype=torch.float32, device=dev)
    R = torch.as_tensor((Rtr - rm) / rs, dtype=torch.float32, device=dev)
    n = len(X); nv = n // 10
    perm = torch.randperm(n, device=dev); va, tr = perm[:nv], perm[nv:]
    net = ErrorNet(X.shape[1], R.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = 1e9, None
    for s in range(steps):
        i = tr[torch.randint(0, len(tr), (bs,), device=dev)]
        mu, lv = net(X[i])
        loss = (0.5 * (lv + (R[i] - mu) ** 2 / lv.exp())).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 99:
            with torch.no_grad():
                mu, lv = net(X[va]); vl = float((0.5 * (lv + (R[va] - mu) ** 2 / lv.exp())).mean())
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    def predict(Xq):
        with torch.no_grad():
            mu, lv = net(torch.as_tensor((Xq - xm) / xs, dtype=torch.float32, device=dev))
        return (mu.cpu().numpy() * rs + rm), (np.exp(0.5 * lv.cpu().numpy()) * rs)
    return predict, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--ds", default="4 8")
    ap.add_argument("--K", type=int, default=4)
    a = ap.parse_args()
    dev = "cuda"
    EV.check_precision(autocast_dtype=torch.float16)
    ad, meta = load_adapter(a.ckpt, dev)
    td = EV.task_data(meta["task"]); A = td.act_dim
    ds = [int(x) for x in a.ds.replace(",", " ").split()]
    dmax = max(ds)
    t0 = time.time()
    etr, ttr = train_windows(td, a.n_train, dmax, seed=5)
    Ztr, ACTtr = [], []
    for i in range(0, a.n_train, 128):
        z, act = filtered(ad, td, etr[i:i + 128], ttr[i:i + 128], dmax, True)
        Ztr.append(z); ACTtr.append(act)
    Ztr, ACTtr = np.concatenate(Ztr), np.concatenate(ACTtr)
    e, t = windows(td, a.n_test, dmax, seed=1)
    Z, ACT = filtered(ad, td, e, t, dmax, True)
    print(f"encoded {a.n_train} train and {a.n_test} test windows ({time.time() - t0:.0f}s)", flush=True)
    cfg = MPPIConfig(population=512, iterations=6)
    make_cost = make_cost_factory(ad, 512, dev, True)
    make_cost1 = make_cost_factory(ad, 1, dev, True)

    def plan(cost_fn, n, seed):
        p = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(seed), dev)
        return p(cost_fn, torch.zeros((n, H, A), dtype=torch.float64, device=dev))

    def hedged_cost(anchors):                                   # anchors (n, K, D)
        n, K, D = anchors.shape
        Zk = torch.as_tensor(anchors, dtype=torch.float32, device=dev)

        def cost(seq):
            s_ = seq.reshape(n, 1, 512, *seq.shape[1:]).expand(n, K, 512, *seq.shape[1:])
            z_ = Zk[:, :, None].expand(n, K, 512, D)
            with torch.autocast("cuda", dtype=torch.float16):
                J = ad.get_cost(z_.reshape(-1, D), s_.reshape(-1, *seq.shape[1:]).float())
            return torch.as_tensor(J, dtype=torch.float64).reshape(n, K, 512).mean(1).reshape(-1)
        return cost

    out = {"meta": meta, "n_train": a.n_train, "n_test": a.n_test, "K": a.K, "by_d": {}}
    for d in ds:
        zh_tr = np.asarray(ad.rollout(Ztr[:, 0], ACTtr[:, :d]), np.float64)[:, -1]
        Rtr = Ztr[:, d] - zh_tr
        Xtr = features(zh_tr, Ztr[:, 0], ACTtr[:, :d])
        predict, vnll = fit(Xtr, Rtr, dev)
        z_t, z_true = Z[:, 0], Z[:, d]
        zh = np.asarray(ad.rollout(z_t, ACT[:, :d]), np.float64)[:, -1]
        mu, sd = predict(features(zh, z_t, ACT[:, :d]))
        err0 = ((z_true - zh) ** 2).sum(1); err1 = ((z_true - zh - mu) ** 2).sum(1)
        cover = float(np.mean(np.abs(z_true - zh - mu) <= 1.96 * sd))
        seed = 2000 + d
        c_true = make_cost1(z_true)
        mu_true = plan(make_cost(z_true), a.n_test, seed)
        base = np.asarray(c_true(mu_true), np.float64)
        reg = lambda m: float((np.asarray(c_true(m), np.float64) - base).mean())
        rng = np.random.default_rng(d)
        bank = Rtr[rng.integers(0, len(Rtr), (a.n_test, a.K - 1))]                  # unconditional draws
        unc = np.concatenate([zh[:, None], zh[:, None] + bank], 1)
        eps = rng.standard_normal((a.n_test, a.K - 1, zh.shape[1]))
        cnd = np.concatenate([(zh + mu)[:, None], (zh + mu)[:, None] + eps * sd[:, None]], 1)
        r = {"err_left": float(err1.mean() / err0.mean()), "val_nll": vnll, "coverage95": cover,
             "regret_pred": reg(plan(make_cost(zh), a.n_test, seed)),
             "regret_corr": reg(plan(make_cost(zh + mu), a.n_test, seed)),
             "regret_uncond4": reg(plan(hedged_cost(unc), a.n_test, seed)),
             "regret_cond4": reg(plan(hedged_cost(cnd), a.n_test, seed)),
             "regret_noise": reg(plan(make_cost(z_true), a.n_test, seed + 7))}
        out["by_d"][str(d)] = r
        print(f"{meta['family']}/{meta['task']} d={d}: error left after correction {r['err_left']:.2f}, "
              f"95% coverage {cover:.2f} | regret pred {r['regret_pred']:.2f} corr {r['regret_corr']:.2f} "
              f"uncond4 {r['regret_uncond4']:.2f} cond4 {r['regret_cond4']:.2f} noise {r['regret_noise']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    od = RESULTS / "latency" / "error_model"; od.mkdir(parents=True, exist_ok=True)
    (od / f"{meta['family']}_{meta['task']}_{meta['run']}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
