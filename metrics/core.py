"""Offline metric implementations (plan Section 6).

These are the *same* functions that run on DreamerV3, TD-MPC2, LeWM and the
diffusion family.  They talk to a model only through `wm.protocol.BaseAdapter`
and to the environment only through `state()` / `set_state()`, so the
linear-Gaussian reference in `metrics/lqg.py` is a legal input and gives every
one of them a closed-form value to be tested against.

Rollout protocols (plan Section 7) are unified as free-running prediction with
a **resynchronisation interval**, which is how the source study
(arXiv 2609.02811) actually framed the difference:

    P1 teacher-forced   resync_every = 1     (measurement update every step)
    P2 free-running     resync_every = inf   (no measurement update)
    P3 interval-matched resync_every = H_use (update on the use's schedule)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

INF = 10 ** 9


@dataclass
class SegmentBatch:
    """Fixed evaluation segments S_M drawn from D_eval (plan Section 2.5)."""
    obs: np.ndarray        # (N, T+1, *obs_shape)
    actions: np.ndarray    # (N, T, A)
    states: np.ndarray     # (N, T+1, S)   ground truth
    rewards: Optional[np.ndarray] = None   # (N, T)

    @property
    def n(self): return self.obs.shape[0]

    @property
    def horizon(self): return self.actions.shape[1]


# ---------------------------------------------------------------------------
# rollout core
# ---------------------------------------------------------------------------
def predict_with_resync(adapter, batch: SegmentBatch, resync_every: int,
                        horizon: Optional[int] = None) -> np.ndarray:
    """Predicted latents under a resynchronisation schedule.

    Returns (N, T+1, D).  Entry [:, t] is the model's prediction of the latent
    at time t.  Index 0 is the encoded truth.  At every step that is a multiple
    of `resync_every` the model is re-initialised from the encoded true
    observation (a 'measurement update'); between those it free-runs.
    """
    T = batch.horizon if horizon is None else horizon
    N = batch.n
    # encoded truth; recurrent families filter the whole prefix
    z_true = np.asarray(adapter.encode_sequence(batch.obs[:, :T + 1],
                                                batch.actions[:, :T]))  # (N, T+1, D)
    D = z_true.shape[-1]
    out = np.empty((N, T + 1, D))
    out[:, 0] = z_true[:, 0]
    z = z_true[:, 0]
    for t in range(T):
        if t > 0 and resync_every > 0 and t % resync_every == 0:
            z = z_true[:, t]
        z = np.asarray(adapter.step(z, batch.actions[:, t]))
        out[:, t + 1] = z
    return out, z_true


def _space(pred, true, adapter, space):
    """Measure the error in raw latent, variance-normalised, or probe space
    (plan Section 6.1, M4/M5 variants (a)/(b)/(c))."""
    if space == "raw":
        return pred, true, 1.0
    if space == "normalised":
        v = true.reshape(-1, true.shape[-1]).var(axis=0).sum()
        return pred, true, max(v, 1e-12)
    if space == "probe":
        sh = pred.shape[:-1]
        p = np.asarray(adapter.probe(pred.reshape(-1, pred.shape[-1]))).reshape(*sh, -1)
        t = np.asarray(adapter.probe(true.reshape(-1, true.shape[-1]))).reshape(*sh, -1)
        return p, t, 1.0
    raise ValueError(space)


def latent_error(adapter, batch, lag=1, resync_every=1, space="raw",
                 use_true_state=False):
    """M4 (lag=1) and M5 (lag=k): mean squared prediction error.

    `use_true_state=True` compares the probe-decoded prediction against the
    environment's true state rather than against the encoded observation,
    which is what M8 needs.
    """
    pred, z_true = predict_with_resync(adapter, batch, resync_every)
    T = batch.horizon
    idx = np.arange(lag, T + 1)
    if use_true_state:
        flat = np.asarray(adapter.probe(pred[:, idx].reshape(-1, pred.shape[-1])))
        p = flat.reshape(batch.n, len(idx), -1)
        t = batch.states[:, idx]
        denom = 1.0
    else:
        p, t, denom = _space(pred[:, idx], z_true[:, idx], adapter, space)
    per_item = ((p - t) ** 2).sum(-1) / denom      # (N, len(idx))
    return per_item


def multistep_error_at(adapter, batch, k, resync_every=INF, space="raw"):
    """Error exactly at horizon k of a rollout started at t=0 (M5).
    Used where the analytic reference is defined per-horizon."""
    pred, z_true = predict_with_resync(adapter, batch, resync_every, horizon=k)
    p, t, denom = _space(pred[:, k:k + 1], z_true[:, k:k + 1], adapter, space)
    return ((p - t) ** 2).sum(-1)[:, 0] / denom


def probe_accuracy(adapter, batch, k, resync_every=INF):
    """M8: R^2 of the probe on *predicted* latents at horizon k against the
    true state.  Total R^2 = 1 - MSE / Var(state)."""
    pred, _ = predict_with_resync(adapter, batch, resync_every, horizon=k)
    p = np.asarray(adapter.probe(pred[:, k]))
    tgt = batch.states[:, k]
    mse = ((p - tgt) ** 2).sum(-1).mean()
    var = tgt.var(axis=0).sum()
    return 1.0 - mse / max(var, 1e-12)


# ---------------------------------------------------------------------------
# M7 matched-intervention audit  (arXiv 2608.29998)
# ---------------------------------------------------------------------------
def matched_intervention(adapter, system, states, actions_a, actions_b, k,
                         g=None):
    """Apply two action sequences that share a root state and differ only in
    the first action, to the environment and to the model, and compare the
    k-step effect.  Process noise cancels because both true rollouts use the
    same seed / are run deterministically from `set_state` (plan Section 6.1).

    Returns dict(gain, r2, cosine) — the source's three propagation numbers.
    """
    sim_a = system.rollout(states, actions_a, noise=False)[:, k]
    sim_b = system.rollout(states, actions_b, noise=False)[:, k]
    z0 = np.asarray(adapter.encode(states))
    wm_a = np.asarray(adapter.rollout(z0, actions_a))[:, k]
    wm_b = np.asarray(adapter.rollout(z0, actions_b))[:, k]
    wm_a = np.asarray(adapter.probe(wm_a)); wm_b = np.asarray(adapter.probe(wm_b))
    d_sim, d_wm = sim_a - sim_b, wm_a - wm_b
    if g is not None:
        g = np.asarray(g).reshape(-1)
        d_sim, d_wm = d_sim @ g, d_wm @ g
        cross = float(np.mean(d_wm * d_sim))
        vw, vs = float(np.mean(d_wm ** 2)), float(np.mean(d_sim ** 2))
    else:
        cross = float(np.mean((d_wm * d_sim).sum(-1)))
        vw = float(np.mean((d_wm ** 2).sum(-1)))
        vs = float(np.mean((d_sim ** 2).sum(-1)))
    return {"gain": cross / vs, "r2": cross ** 2 / (vw * vs),
            "cosine": cross / np.sqrt(vw * vs)}


# ---------------------------------------------------------------------------
# M9 disagreement
# ---------------------------------------------------------------------------
def sample_disagreement(adapter, batch, k, n_samples=8):
    """M9 for stochastic families: mean variance across samples of the
    predicted latent at horizon k."""
    z = np.asarray(adapter.encode(batch.obs[:, 0]))
    draws = []
    for _ in range(n_samples):
        zz = z
        for t in range(k):
            zz = np.asarray(adapter.sample(zz, batch.actions[:, t], 1))[0]
        draws.append(zz)
    return float(np.stack(draws).var(axis=0).sum(-1).mean())


def ensemble_disagreement(adapters, batch, k):
    """M9 for deterministic families: variance across seed-matched
    checkpoints (plan Section 6.1, M9)."""
    preds = []
    for ad in adapters:
        z = np.asarray(ad.encode(batch.obs[:, 0]))
        preds.append(np.asarray(ad.rollout(z, batch.actions[:, :k]))[:, k])
    return float(np.stack(preds).var(axis=0).sum(-1).mean())


# ---------------------------------------------------------------------------
# M10 reward observability fraction  (arXiv 2607.01736, eq. 7)
# ---------------------------------------------------------------------------
def observable_subspace(A, C, tol=1e-9):
    """Orthonormal basis of the observable subspace of the pair (A, C)."""
    A, C = np.asarray(A), np.atleast_2d(C)
    n = A.shape[0]
    O = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])
    U, s, Vt = np.linalg.svd(O)
    r = int((s > tol * max(1.0, s[0])).sum())
    return Vt[:r].T                     # (n, r), columns orthonormal


def reward_observability_fraction(A, C, r_grad):
    """ROF = ||V_o^T r||^2 / ||r||^2.  Fraction of reward-gradient energy that
    lies in the observable subspace of the model's latent dynamics."""
    V = observable_subspace(A, C)
    r = np.asarray(r_grad).reshape(-1)
    if V.shape[1] == 0:
        return 0.0
    return float(np.sum((V.T @ r) ** 2) / max(np.sum(r ** 2), 1e-12))


def jacobians_from_adapter(adapter, z0, a0, eps=1e-5):
    """Finite-difference Jacobians so ROF works for any adapter, not only
    linear ones."""
    z0 = np.atleast_2d(z0)[0]; a0 = np.atleast_2d(a0)[0]
    D = z0.size
    A = np.zeros((D, D))
    for i in range(D):
        dz = np.zeros(D); dz[i] = eps
        zp = adapter.step((z0 + dz)[None], a0[None])[0]
        zm = adapter.step((z0 - dz)[None], a0[None])[0]
        A[:, i] = (zp - zm) / (2 * eps)
    try:
        r_grad = np.zeros(D)
        for i in range(D):
            dz = np.zeros(D); dz[i] = eps
            rp = float(np.atleast_1d(adapter.reward((z0 + dz)[None]))[0])
            rm = float(np.atleast_1d(adapter.reward((z0 - dz)[None]))[0])
            r_grad[i] = (rp - rm) / (2 * eps)
    except NotImplementedError:
        r_grad = np.ones(D)
    return A, r_grad


# ---------------------------------------------------------------------------
# M11 plan-cost discrepancy  (arXiv 2607.10362)
# ---------------------------------------------------------------------------
def plan_cost_discrepancy(adapter, system, states, candidate_actions):
    """g = sup_u |J_hat(u) - J(u)| over the planner's candidate set, plus the
    mean discrepancy and 1 - Spearman(J_hat, J).

    `candidate_actions` is (N, U, H, A): U candidates per start state.
    """
    from scipy.stats import spearmanr
    N, U, H, A = candidate_actions.shape
    sup, mean_abs, rhos = [], [], []
    for i in range(N):
        s0 = np.repeat(states[i][None], U, axis=0)
        J = system.cost(s0, candidate_actions[i])
        z0 = np.asarray(adapter.encode(s0))
        Jh = adapter.cost(z0, candidate_actions[i]) if hasattr(adapter, "cost") \
            else _model_cost(adapter, z0, candidate_actions[i], system)
        d = np.abs(Jh - J)
        sup.append(d.max()); mean_abs.append(d.mean())
        if np.std(J) > 1e-12 and np.std(Jh) > 1e-12:
            rhos.append(spearmanr(Jh, J).statistic)
    return {"g_sup": float(np.mean(sup)), "mean_abs": float(np.mean(mean_abs)),
            "one_minus_rho": float(1.0 - np.nanmean(rhos)) if rhos else np.nan}


def _model_cost(adapter, z0, actions, system):
    traj = np.asarray(adapter.rollout(z0, actions))
    st = np.asarray(adapter.probe(traj[:, 1:].reshape(-1, traj.shape[-1])))
    st = st.reshape(traj.shape[0], traj.shape[1] - 1, -1)
    c = np.einsum("bti,ij,btj->b", st, system.Q, st)
    c += np.einsum("bti,ij,btj->b", actions, system.R, actions)
    return c


# ---------------------------------------------------------------------------
# M12 iKCE  (arXiv 2607.05966)
# ---------------------------------------------------------------------------
def ikce(adapter, batch, T, K):
    """Kinematic-consistency error of the model's own imagined trajectory:
    mean_t ||x_hat_{t+1} - kin(x_hat_t)||^2 with `kin` the linear
    constant-velocity null model K (acting in probe space)."""
    z0 = np.asarray(adapter.encode(batch.obs[:, 0]))
    traj = np.asarray(adapter.rollout(z0, batch.actions[:, :T]))
    x = np.asarray(adapter.probe(traj.reshape(-1, traj.shape[-1]))).reshape(
        traj.shape[0], T + 1, -1)
    resid = x[:, 1:] - x[:, :-1] @ np.asarray(K).T
    return float((resid ** 2).sum(-1).mean())
