"""Residual-hedged committed-prefix planning (side paper, method candidate).

Committed-prefix planning scores every candidate from one predicted post-delay
latent z_hat = rollout(z(t), committed).  Here each candidate's cost is the
mean over K anchors z_hat + r_k, where r_1 = 0 and r_2..r_K are residuals of
the same model's own d-step predictions, z(t+d) - rollout(z(t), a(t:t+d)),
measured on training windows before any episode (a calibration bank; no
ensemble, no retraining).  The prefix is rolled once per episode, and the
candidates' free part is scored from every anchor with the same samples.
With K = 1 this is committed-prefix planning in its roll-once form.

Calibration uses D_train episodes only (0 .. N_TRAIN - 1); the side paper's
episodes use seeds 950,000+ in the simulator, and the diagnostics used D_eval.
"""
import numpy as np
import torch

from experiments.dmc import evaluate as EV

CTX = 16


def residual_bank(ad, task, d, n=512, seed=11, H=32):
    """(n, D) residuals z(t+d) - rollout(z(t), a(t:t+d)) on D_train windows."""
    td = EV.task_data(task)
    rng = np.random.default_rng(seed + d)
    e = rng.integers(0, EV.N_TRAIN, n)
    t = rng.integers(CTX + 8, td.L - d - 2, n)
    out = []
    for i in range(0, n, 64):
        ei, ti = e[i:i + 64], t[i:i + 64]
        obs = np.stack([td.frames[a, b - CTX:b + d + 1] for a, b in zip(ei, ti)])
        act = np.stack([td.action[a, b - CTX:b + d] for a, b in zip(ei, ti)])
        z = np.asarray(ad.encode_sequence(obs, act), np.float64)[:, CTX:]
        zh = np.asarray(ad.rollout(z[:, 0], act[:, CTX:CTX + d]), np.float64)[:, -1]
        out.append(z[:, d] - zh)
    return np.concatenate(out)


def make_cost_hedged(ad, P, dev, amp, d, bank, K, H, gen_seed=5, batch_anchors=False, max_width=51_200,
                     anchor=None):
    """make_cost(Z) for the cpp schedule: the cost function receives candidates
    [prefix (d); free (H)] and returns the anchor-averaged cost of the free part.
    The pre-episode plan (candidates of length H, no prefix, simulator paused)
    is scored from Z alone.

    batch_anchors=False scores the anchors one after another (the pilot and its
    replication).  batch_anchors=True scores all K anchors in as few get_cost
    calls as fit max_width candidate rows, which is what makes hedging cheap at
    batch size 1: single-environment latency is set by the H sequential model
    steps, not by the number of rows.  Same per-row computation either way."""
    rng = np.random.default_rng(gen_seed)
    bank_t = torch.as_tensor(bank, dtype=torch.float32, device=dev)
    anchor = anchor or (lambda zh, r: zh + r)          # Section 18: log-space anchors for SimNorm latents

    def make_cost(Z):
        Z = np.asarray(Z)
        E = len(Z)

        def cost(seq):
            L, A = seq.shape[1], seq.shape[2]
            if L == H:                                                  # pre-episode plan: state known
                Zr = torch.as_tensor(Z, dtype=torch.float32, device=dev).repeat_interleave(P, 0)
                with torch.autocast("cuda", dtype=torch.float16) if amp else torch.autocast("cpu", enabled=False):
                    return torch.as_tensor(ad.get_cost(Zr, seq.float()), dtype=torch.float64, device=dev)
            assert L == H + d
            s = seq.reshape(E, P, L, A)
            prefix = s[:, 0, :d].float().cpu().numpy()
            zh = torch.as_tensor(np.asarray(ad.rollout(Z, prefix))[:, -1], dtype=torch.float32, device=dev)
            free = s[:, :, d:].float().reshape(E * P, L - d, A)
            idx = rng.integers(0, len(bank), (E, max(K - 1, 1)))         # per-episode draws
            J = torch.zeros(E * P, dtype=torch.float64, device=dev)
            if batch_anchors:
                anc = torch.stack([zh] + [anchor(zh, bank_t[idx[:, k - 1]]) for k in range(1, K)], 0)   # (K, E, D)
                Zall = anc.repeat_interleave(P, 1).reshape(K * E * P, -1)
                Fall = free.unsqueeze(0).expand(K, *free.shape).reshape(K * E * P, *free.shape[1:])
                Js = []
                for i in range(0, len(Zall), max_width):
                    with torch.autocast("cuda", dtype=torch.float16) if amp else torch.autocast("cpu", enabled=False):
                        Js.append(torch.as_tensor(ad.get_cost(Zall[i:i + max_width], Fall[i:i + max_width]),
                                                  dtype=torch.float64, device=dev))
                return torch.cat(Js).reshape(K, E * P).mean(0)
            for k in range(K):
                anc = zh if k == 0 else anchor(zh, bank_t[idx[:, k - 1]])
                Zr = anc.repeat_interleave(P, 0)
                with torch.autocast("cuda", dtype=torch.float16) if amp else torch.autocast("cpu", enabled=False):
                    Jk = ad.get_cost(Zr, free)
                J += torch.as_tensor(Jk, dtype=torch.float64, device=dev)
            return J / K
        return cost
    return make_cost


# --- Section 18: residuals in the coordinates of a SimNorm latent (TD-MPC2) ---------------------
GROUP, EPS = 8, 1e-8


def group_log(z):
    return torch.log(torch.as_tensor(z).clamp_min(EPS))


def group_softmax(x):
    sh = x.shape
    return torch.softmax(x.reshape(*sh[:-1], -1, GROUP), -1).reshape(sh)


def log_anchor(zh, r):
    """softmax_g(log z_hat + r): an anchor on the simplex of each SimNorm group."""
    return group_softmax(group_log(zh) + r)


def residual_bank_log(ad, task, d, n=512, seed=11):
    """As residual_bank, with residuals log z(t+d) - log rollout(z(t), a(t:t+d)) per unit."""
    td = EV.task_data(task)
    rng = np.random.default_rng(seed + d)
    e = rng.integers(0, EV.N_TRAIN, n)
    t = rng.integers(CTX + 8, td.L - d - 2, n)
    out = []
    for i in range(0, n, 64):
        ei, ti = e[i:i + 64], t[i:i + 64]
        obs = np.stack([td.frames[a, b - CTX:b + d + 1] for a, b in zip(ei, ti)])
        act = np.stack([td.action[a, b - CTX:b + d] for a, b in zip(ei, ti)])
        z = np.asarray(ad.encode_sequence(obs, act), np.float64)[:, CTX:]
        zh = np.asarray(ad.rollout(z[:, 0], act[:, CTX:CTX + d]), np.float64)[:, -1]
        out.append((group_log(torch.as_tensor(z[:, d])) - group_log(torch.as_tensor(zh))).numpy())
    return np.concatenate(out)
