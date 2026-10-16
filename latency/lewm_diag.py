"""Study 16C, exploratory part (EXPERIMENT_PLAN_SIDE.md Section 16C): the offline hedging value V for
the LeWM models, whose closed-loop results (Study 15B) are known.

On 64 dataset windows from episodes not used for evaluation, with the recorded actions as the committed
prefix of d blocks: z_t = enc(o_t), the true post-delay embedding z = enc(o_{t+5d}), the prediction
z_hat = rollout(z_t, prefix), and the goal enc(o_{t+5d+25}) as in the hand-over setting.  LeWM's CEM
plans H = 5 free blocks from z (reference), from z_hat (committed prefix), from z_t as if no time had
passed (stale), and from four anchors z_hat + residuals (hedged, residual bank of latency/lewm.py), all
with the same sampling noise.  Each plan's free blocks are scored with LeWM's cost from z, minus the
reference plan's cost; V = (regret_pred - regret_hedge_K4) / regret_naive, averaged over windows.
Writes results/latency/lewm/diag16.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import lewm as L  # noqa: E402


def main():
    dev = "cuda"
    out = {}
    for task in ("pusht", "cube", "reacher"):
        c = L.TASKS[task]
        model = L.load_model(task, dev)
        ds = L.load_dataset(task)
        process = L.processors(ds, c["keys"])
        col, ep, step, Lall = L.episode_lengths(ds)
        out[task] = {}
        for d, g_off in ((2, 35), (4, 45)):
            eval_eps, _ = L.eval_starts(ds, 200, 951_000, g_off)
            bank = torch.as_tensor(L.residual_bank(model, ds, task, d, np.unique(eval_eps), process, dev),
                                   dtype=torch.float32, device=dev)
            rng = np.random.default_rng(16_000 + d)
            ids = np.setdiff1d(np.unique(ep), np.unique(eval_eps))
            ids = ids[Lall[ids] > L.BLOCK * d + 26 + 1]
            e = rng.choice(ids, 64, replace=False)
            t = np.array([rng.integers(0, Lall[x] - L.BLOCK * d - 26) for x in e])
            chunks = ds.load_chunk(e, t, t + L.BLOCK * d + 26)
            tf = L.img_transform()
            regs = {k: [] for k in ("pred", "naive", "hedge", "noise")}
            for ch in chunks:
                a = ch["action"][:L.BLOCK * d]
                a = a.numpy() if torch.is_tensor(a) else np.asarray(a)
                if np.isnan(a).any():
                    continue
                pre = torch.as_tensor(process["action"].transform(a).reshape(1, d, -1), dtype=torch.float32, device=dev)
                px = ch["pixels"]
                z_t = L.encode(model, tf(px[0])[None].to(dev))
                z = L.encode(model, tf(px[L.BLOCK * d])[None].to(dev))
                goal = L.encode(model, tf(px[-1])[None].to(dev))
                zh = L.rollout(model, z_t[:, None], pre)[:, -1]
                A = pre.shape[-1]
                zero = torch.zeros((1, L.H, A), device=dev)

                def plan(z0, anchors=None, seed=1):
                    gen = torch.Generator(device=dev).manual_seed(seed)
                    if anchors is None:
                        return L.cem(L.cost_fn(model, z0, goal), zero, gen, dev)
                    # hedged: cost averaged over anchors z0 + r (no prefix: anchors are post-delay states)
                    def cost(cand):
                        return torch.stack([L.cost_fn(model, z0 + anchors[:, k], goal)(cand)
                                            for k in range(anchors.shape[1])]).mean(0)
                    return L.cem(cost, zero, gen, dev)

                def J(mu):
                    return float(L.cost_fn(model, z, goal)(mu[:, None])[0, 0])
                ref = J(plan(z))
                idx = torch.as_tensor(rng.integers(0, len(bank), 3), device=dev)
                anchors = torch.cat([torch.zeros_like(zh)[:, None], bank[idx][None]], 1)
                regs["pred"].append(J(plan(zh)) - ref)
                regs["naive"].append(J(plan(z_t)) - ref)
                regs["hedge"].append(J(plan(zh, anchors)) - ref)
                regs["noise"].append(J(plan(z, seed=2)) - ref)
            r = {k: float(np.mean(v)) for k, v in regs.items()}
            r["V"] = (r["pred"] - r["hedge"]) / max(r["naive"], 1e-9)
            r["windows"] = len(regs["pred"])
            out[task][f"d{d}"] = r
            print(task, d, {k: round(v, 4) for k, v in r.items()}, flush=True)
    (L.OUT / "diag16.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
