"""Diagnostic for the residual delay mismatch (side paper, method design; not a
paper result and not a model-quality measure across checkpoints).

Committed-prefix planning starts the plan from the model's prediction of the
post-delay latent, z_hat(t+d) = rollout(z(t), a(t:t+d)), instead of the filtered
latent z(t+d) it would have with a paused simulator.  On held-out D_eval windows
(recorded actions as the committed prefix) this measures, per delay d:

  anchor error     |z_hat - z|^2 relative to the stale error |z(t) - z(t+d)|^2
  plan error       first-d actions of MPPI from z_hat vs from z, same sampling
                   noise (common random numbers); the noise floor is the same
                   comparison from z with different noise
  anchor regret    model-predicted cost of each plan, evaluated from the true z,
                   minus that of the plan made from z (>= 0 up to MPPI noise)
and how much of the plan error / regret two candidate corrections remove:

  C1 sensitivity   finite differences of the plan along the top-p principal
                   directions of calibration residuals (z - z_hat on separate
                   windows), first-order correction with the true z at reveal
  C2 multi-anchor  K anchors z_hat + calibration residual draws; the plan of the
                   anchor nearest to the true z (whitened by the residual PCA)

Writes results/latency/diagnose/<family>_<task>_<run>.json.
"""
import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.dmc import evaluate as EV  # noqa: E402
from latency.run_dmc import H, load_adapter, make_cost_factory  # noqa: E402
from latency.schedule import MPPIConfig, PrefixMPPI  # noqa: E402
from paths import RESULTS  # noqa: E402

CTX = 16


def windows(td, n, d_max, seed):
    rng = np.random.default_rng(seed)
    e = rng.integers(EV.N_TRAIN, EV.N_EP, n)
    t = rng.integers(CTX + 8, td.L - H - d_max - 2, n)
    return e, t


def filtered(ad, td, e, t, d_max, rec):
    """Filtered latents z(t) .. z(t + d_max) for each window, with a burn-in of CTX frames."""
    obs = np.stack([td.frames[ei, ti - CTX:ti + d_max + 1] for ei, ti in zip(e, t)])
    act = np.stack([td.action[ei, ti - CTX:ti + d_max] for ei, ti in zip(e, t)])
    z = ad.encode_sequence(obs, act)[:, CTX:]
    return np.asarray(z, np.float64), act[:, CTX:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=64, help="test windows")
    ap.add_argument("--n-cal", type=int, default=256, help="calibration windows")
    ap.add_argument("--p", type=int, default=8, help="C1 directions")
    ap.add_argument("--K", type=int, default=16, help="C2 anchors")
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--ds", default="1,2,4,8")
    ap.add_argument("--tag", default="", help="suffix of the output file (Section 16C: _v16)")
    ap.add_argument("--extra", action="store_true",
                    help="mechanism tests only: random / shuffled errors of the same size, DATS-style hedging")
    a = ap.parse_args()
    dev = "cuda"
    EV.check_precision(autocast_dtype=torch.float16)
    ad, meta = load_adapter(a.ckpt, dev)
    task = meta["task"]
    td = EV.task_data(task)
    A = td.act_dim
    rec = meta["family"] in ("rssm", "tdmpc2") or meta.get("arch") == "hist3"
    ds = [int(x) for x in a.ds.split(",")]
    dmax = max(ds)
    e, t = windows(td, a.n, dmax, seed=1)
    ec, tc = windows(td, a.n_cal, dmax, seed=2)
    Z, ACT = filtered(ad, td, e, t, dmax, rec)
    Zc, ACTc = filtered(ad, td, ec, tc, dmax, rec)
    cfg = MPPIConfig(population=a.pop, iterations=a.iters)
    make_cost = make_cost_factory(ad, a.pop, dev, True)
    make_cost1 = make_cost_factory(ad, 1, dev, True)

    def plan(anchor, seed):
        planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(seed), dev)
        warm = torch.zeros((len(anchor), H, A), dtype=torch.float64, device=dev)
        return planner(make_cost(anchor), warm)

    def cost_from(anchor, mu):
        return torch.as_tensor(make_cost1(anchor)(mu), dtype=torch.float64).cpu().numpy()

    out = {"meta": meta, "n": a.n, "n_cal": a.n_cal, "p": a.p, "K": a.K, "pop": a.pop, "iters": a.iters,
           "H": H, "gpu": torch.cuda.get_device_name(0), "by_d": {}}
    t0 = time.time()
    for d in ds:
        z_t, z_true = Z[:, 0], Z[:, d]
        z_hat = np.asarray(ad.rollout(z_t, ACT[:, :d]), np.float64)[:, -1]
        zc_hat = np.asarray(ad.rollout(Zc[:, 0], ACTc[:, :d]), np.float64)[:, -1]
        res_cal = Zc[:, d] - zc_hat                                     # calibration residuals
        mu_r = res_cal.mean(0)
        U, S, Vt = np.linalg.svd(res_cal - mu_r, full_matrices=False)
        sig = S / np.sqrt(len(res_cal))
        P = Vt[:a.p]                                                    # (p, D)
        err = z_true - z_hat
        r = {}
        r["anchor_err"] = float((err ** 2).sum(1).mean())
        r["stale_err"] = float(((z_true - z_t) ** 2).sum(1).mean())
        r["anchor_err_frac_in_top_p"] = float(((err @ P.T) ** 2).sum(1).mean() / max(r["anchor_err"], 1e-12))
        r["cal_var_frac_top_p"] = float((sig[:a.p] ** 2).sum() / max((sig ** 2).sum(), 1e-12))
        seed = 1000 + d
        mu_true = plan(z_true, seed)
        mu_true2 = plan(z_true, seed + 7)                               # noise floor
        mu_hat = plan(z_hat, seed)
        sl = slice(0, d)
        pe = lambda m: float(((m[:, sl] - mu_true[:, sl]) ** 2).sum((1, 2)).mean().item())
        c_true = cost_from(z_true, mu_true)
        reg = lambda m: float((cost_from(z_true, m) - c_true).mean())
        r["plan_err_pred"] = pe(mu_hat); r["plan_err_noise"] = pe(mu_true2)
        r["regret_pred"] = reg(mu_hat); r["regret_noise"] = reg(mu_true2)
        r["regret_naive"] = reg(plan(z_t, seed))                       # plan from the stale latent, as if current
        if a.extra:
            rng = np.random.default_rng(100 + d)
            nrm = np.linalg.norm(err, axis=1, keepdims=True)
            g = rng.standard_normal(err.shape); g *= nrm / np.linalg.norm(g, axis=1, keepdims=True)
            r["regret_rand_dir"] = reg(plan(z_true + g, seed))            # same size, random direction
            sh = res_cal[rng.integers(0, len(res_cal), len(err))]
            r["regret_shuffled_err"] = reg(plan(z_true + sh, seed))       # a real residual from another window
            r["regret_half_err"] = reg(plan(z_true - 0.5 * err, seed))    # halfway between z_hat and z
            for K in (4, 16):                                             # DATS-style: one plan, cost averaged
                anc = np.stack([z_hat] + [z_hat + res_cal[rng.integers(0, len(res_cal), len(err))]
                                          for _ in range(K - 1)], 1)       # (n, K, D)
                Zk = torch.as_tensor(anc, dtype=torch.float32, device=dev)

                def cost_avg(seq, Zk=Zk, K=K):
                    n = Zk.shape[0]
                    s_ = seq.reshape(n, 1, a.pop, *seq.shape[1:]).expand(n, K, a.pop, *seq.shape[1:])
                    z_ = Zk[:, :, None].expand(n, K, a.pop, Zk.shape[-1])
                    with torch.autocast("cuda", dtype=torch.float16):
                        J = ad.get_cost(z_.reshape(-1, Zk.shape[-1]), s_.reshape(-1, *seq.shape[1:]).float())
                    return torch.as_tensor(J, dtype=torch.float64).reshape(n, K, a.pop).mean(1).reshape(-1)
                planner = PrefixMPPI(cfg, torch.Generator(device=dev).manual_seed(seed), dev)
                mu_avg = planner(cost_avg, torch.zeros((len(err), H, A), dtype=torch.float64, device=dev))
                r[f"regret_hedge_K{K}"] = reg(mu_avg)
            out["by_d"][str(d)] = r
            print(f"{meta['family']}/{task} d={d}: regret naive {r['regret_naive']:.2f} pred {r['regret_pred']:.2f} "
                  f"noise {r['regret_noise']:.2f} | rand-dir {r['regret_rand_dir']:.2f} shuffled {r['regret_shuffled_err']:.2f} "
                  f"half {r['regret_half_err']:.2f} | hedge K4 {r['regret_hedge_K4']:.2f} K16 {r['regret_hedge_K16']:.2f}",
                  flush=True)
            continue
        # C1: central differences along the top-p residual directions, common random numbers
        h = 1.0
        corr = mu_hat.clone()
        coef = (z_true - z_hat) @ P.T                                   # (n, p) true offsets, revealed at t+d
        for j in range(a.p):
            step = h * sig[j] * P[j]
            mp = plan(z_hat + step, seed); mm = plan(z_hat - step, seed)
            sens = (mp - mm) / (2 * h * sig[j])                         # d mu / d (unit offset along P_j)
            corr = corr + sens * torch.as_tensor(coef[:, j], dtype=torch.float64, device=dev)[:, None, None]
        corr = corr.clamp(-1, 1)
        r["plan_err_C1"] = pe(corr); r["regret_C1"] = reg(corr)
        # C1 with bias correction of the anchor (mean calibration residual), no sensitivities
        mu_b = plan(z_hat + mu_r, seed)
        r["plan_err_bias"] = pe(mu_b); r["regret_bias"] = reg(mu_b)
        # C2: K anchors from calibration residuals; nearest in whitened top-p coordinates
        rng = np.random.default_rng(d)
        draws = res_cal[rng.integers(0, len(res_cal), a.K - 1)]
        anchors = [z_hat] + [z_hat + dr for dr in draws]
        W = P / np.maximum(sig[:a.p, None], 1e-12)
        dist = np.stack([(((z_true - an) @ W.T) ** 2).sum(1) for an in anchors], 1)    # (n, K)
        pick = dist.argmin(1)
        mus = torch.stack([plan(an, seed) for an in anchors], 1)        # (n, K, H, A)
        mu_sel = mus[torch.arange(len(pick)), torch.as_tensor(pick, device=dev)]
        r["plan_err_C2"] = pe(mu_sel); r["regret_C2"] = reg(mu_sel)
        r["C2_pick_nominal_frac"] = float((pick == 0).mean())
        out["by_d"][str(d)] = r
        print(f"{meta['family']}/{task} d={d}: anchor/stale err {r['anchor_err'] / max(r['stale_err'], 1e-12):.2f}, "
              f"top-{a.p} share {r['anchor_err_frac_in_top_p']:.2f} | plan err pred {r['plan_err_pred']:.3f} "
              f"noise {r['plan_err_noise']:.3f} C1 {r['plan_err_C1']:.3f} bias {r['plan_err_bias']:.3f} "
              f"C2 {r['plan_err_C2']:.3f} | regret naive {r['regret_naive']:.2f} pred {r['regret_pred']:.2f} "
              f"noise {r['regret_noise']:.2f} C1 {r['regret_C1']:.2f} bias {r['regret_bias']:.2f} "
              f"C2 {r['regret_C2']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    od = RESULTS / "latency" / "diagnose"
    od.mkdir(parents=True, exist_ok=True)
    f = od / f"{meta['family']}_{task}_{meta['run']}{'_extra' if a.extra else ''}{a.tag}.json"
    f.write_text(json.dumps(out, indent=1)); print("wrote", f)


if __name__ == "__main__":
    main()
