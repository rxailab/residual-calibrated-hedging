"""Evaluate one DMControl main-grid checkpoint (pixel RSSM or pixel JEPA):
every offline metric and two closed-loop utilities, written to storage as soon
as they exist.  Structure and metric keys follow experiments/lunar/evaluate.py.

Utilities (plan Section 5)
  U1  closed-loop planning return: MPPI through the world model with the
      Appendix A settings (H_plan 16, replan every decision, population 512,
      6 iterations, warm start), 100 episodes on fixed seeds, 500 decisions
      each.  The episodes run in lockstep and every MPPI iteration is ONE
      batched get_cost call over 100 x 512 candidate sequences.
  U2  policy-ranking agreement: Kendall tau between the imagined and the true
      100-decision returns of 12 fixed policies built on the task's TD-MPC2
      policy prior (experiments/dmc/u2_policies.py).
Metrics (plan Section 6) under protocols P1 / P3 / P2 (plan Section 7), on 500
fixed D_eval segments of 64 decisions.

Design choices and deviations, for the pre-registration log
 1. Pixels.  Every frame a model sees is rendered at 128x128, averaged over
    exact 2x2 blocks and rounded to uint8, the rule of preprocess.py
    (u2_policies.pool).  A start-up check re-renders stored D_eval states and
    compares them with frames.npy; the result is stored with every checkpoint.
 2. Burn-in context.  One pixel frame carries no velocity, so an RSSM latent
    needs history (plan amendment A7).  Every metric segment is preceded by 16
    real frames that the posterior filters through before t = 0, and the P1 /
    P3 resynchronisations use that same filtered posterior (plan Section 7, P2:
    "condition on the ground-truth context window").  The probe is fit on
    latents with the same burn-in.  The JEPA encoder is Markovian and ignores it.
 3. State.  The probe's qpos/qvel columns are the state of every probe-space
    metric, with two corrections for what a camera can see.  A hinge without
    joint limits winds (cartpole's pole spans -39..44 rad in D, walker's torso
    pitch -9..15 rad) and a frame shows it only modulo 2 pi, so it enters as
    (cos, sin).  Walker's horizontal root position is dropped: camera 0 tracks
    the centre of mass and the coordinate grows without bound (-8..45 m); its
    velocity stays.  Every probe column is z-scored with D_train statistics, so
    probe-space errors weight all coordinates equally instead of being
    dominated by joint velocities (walker's are 10-13 rad/s s.d.).
 4. M10.  The observability matrix C is the probe's linear map onto the state
    columns, for both families (the RSSM decoder Jacobian is 12288 x 544).
    Row scaling does not change the observable subspace, so z-scoring does not
    matter.  Everything else is the LunarLander computation, with the K-step
    horizon K = H_plan = 16.
 5. M12.  Constant-velocity null model fit per joint on real D_train
    transitions (qpos_i' = qpos_i + c_i qvel_i; every joint of the three tasks
    is a slide or a hinge), on probe-decoded states in physical units; for a
    winding hinge the position residual is the wrapped angle difference.  All
    joints count, so on walker the residual is dominated by the actuated
    joints' velocity changes (real value about 1300 against 0.06 on cartpole);
    the plan's wording ("root position and velocity") would keep only the root.
 6. set_state replays (M7, M11) reset dm_control's step counter: after its
    time limit the environment silently resets itself on the next step, which
    would corrupt the later replays from a root.
 7. U1 planner.  The MPPI update of planners/mppi.py (softmax weights,
    momentum, minimum std, warm start), batched over episodes, PlanConfig
    defaults except action bounds [-1, 1], the DMC action box.  The planner's
    rollouts run under fp16 autocast (plan amendment A3; never bf16): costs
    rank candidates as fp32 does (within-episode rank correlation 0.9998 or
    better for the RSSM, 0.995 or better for the JEPA; no non-finite values)
    and the plan is 1.6-3.5x faster.  A non-finite cost counts as 1e9
    (planners/mppi.py maps NaN and +inf only).  The simulator side (physics,
    render, pooling) can run in worker processes (--env-workers); frames and
    returns are identical, only the wall time changes.
 8. Units.  obs_rmse is in the decoder's normalised pixel units (pixel / 255);
    probe_rmse and probe-space latent errors are in z-scored state units.
    M9 is measured at k = 16 = H_plan (LunarLander: 10 of 25).
 9. Probe regularisation.  5-fold cross-validation with whole windows held out
    (plan Appendix B), one penalty for all z-scored targets.  The LunarLander
    file's leave-one-out GCV lets neighbouring frames of a window leak into
    each other and picks penalties 3-5 orders of magnitude smaller; those
    probes have the same held-out R^2 on encoded D_eval latents but amplify
    small prior/posterior differences (cartpole RSSM, step 1800: M8 at k = 1
    is -1.02 with GCV, 0.64 with grouped CV).
10. U2.  The policy set is in u2_policies.py (eps ladder to 0.9, since eps = 1
    is the uniform random policy already in the set).  Imagined rollouts start
    from the encoding of each start state's first frame, as in LunarLander.
11. TD-MPC2 (wm/tdmpc2.py).  The latent encodes a stack of 3 frames, so it is
    history-dependent like the RSSM's: segments get the burn-in context, roots
    are encoded from their prefix, and U1 advances each episode's frame stack.
    Its reward head scores (z_t, a_t) before the step, so U2 accumulates
    r(z_t, a_t) and then steps; the other families score the next latent.  No
    decoder (obs_rmse not applicable) and deterministic (M9 not applicable).
"""
import os, sys, pathlib, json, time, argparse, datetime, traceback, dataclasses, contextlib
import multiprocessing as mp
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from wm.pixel import PixelRSSM, PixelJEPA, PixelRSSMAdapter, PixelJEPAAdapter, PixelJEPAHist, PixelJEPAHistAdapter
from wm.tdmpc2 import PixelTDMPC2, TDMPC2Adapter
from wm.protocol import check_precision
from metrics.core import (SegmentBatch, latent_error, predict_with_resync, probe_accuracy,
                          sample_disagreement, INF)
from planners.mppi import PlanConfig
from envs.dmc import DMCEnv
from experiments.dmc import u2_policies as U2P
from experiments.dmc.u2_policies import pool
from paths import RESULTS

DATA = pathlib.Path("/scratch/hpc/11/xiar3/MetricTrust/data/dmc64")
CKPT = pathlib.Path("/scratch/hpc/11/xiar3/MetricTrust/checkpoints/dmc")
PROG = RESULTS / "progress" / "dmc_eval"
N_TRAIN, N_EP = 1900, 2000                    # episodes 1900-1999 are D_eval
TASKS = ("cartpole-swingup", "walker-walk", "cup-catch")
FAMILIES = ("rssm", "jepa", "tdmpc2")
RUNS = ("seed0", "seed1", "seed2", "seed0_d10", "seed0_d30", "seed0_w0.5", "seed0_w2")
FAMILY_RUNS = {"rssm": RUNS, "jepa": RUNS,            # TD-MPC2's size variants are its own 1M and 19M
               "tdmpc2": ("seed0", "seed1", "seed2", "seed0_d10", "seed0_d30", "seed0_m1", "seed0_m19")}
POPW_FRACS = (0.03, 0.06, 0.12, 0.25, 0.5, 1.0)          # plan Section 2.2
BUDGET = {"rssm": 60000, "jepa": 60000, "tdmpc2": 100000}   # tdmpc2: plateau pilot, TDMPC2_BUDGET_PILOT.md
POPW_STEPS = (1800, 3600, 7200, 15000, 30000, 60000)      # 3, 6, 12, 25, 50, 100 % of 60k
U1_SEEDS = list(range(900_000, 900_100))                   # S_U1, shared by all checkpoints
ALPHAS = np.logspace(-2, 6, 9)
# qpos coordinates a camera cannot identify (design note 3), by qpos index
TASK_STATE = {"cartpole-swingup": {"wind": (1,), "drop": ()},   # pole hinge_1 (unlimited)
              "walker-walk": {"wind": (2,), "drop": (1,)},       # rooty (unlimited); rootx (trackcom camera)
              "cup-catch": {"wind": (), "drop": ()}}             # four slides, fixed camera


@dataclasses.dataclass
class Settings:
    seg_T: int = 64          # decisions per metric segment = the P2 horizon
    n_seg: int = 500         # |S_M|
    ctx: int = 16            # burn-in frames before each segment and probe window
    probe_win: int = 400     # D_train windows for the probe
    probe_T: int = 48        # latents per window after the burn-in (49 used)
    n_roots: int = 60        # D_eval root states for M7 and M11 (60: cup-catch's sparse reward left 7 of 20 informative)
    m9_n: int = 200
    m10_n: int = 12
    h_plan: int = 32         # H_plan (amendment A10, pilot: 32 for all three tasks): P3 interval, M9/M11/M12 horizon, M10 K
    pop: int = 512
    iters: int = 6
    u1_episodes: int = 100
    u1_T: int = 500
    u2_seeds: int = 50
    u2_H: int = 100
    fake_render: bool = False  # CPU smoke tests only (the login node cannot render)
    chunk: int = 50            # segments per encoder call


TINY = dict(h_plan=8, seg_T=20, n_seg=12, ctx=4, probe_win=12, probe_T=10, n_roots=2, m9_n=6, m10_n=2,
            pop=16, u1_episodes=2, u1_T=4, u2_seeds=3, u2_H=5, fake_render=True, chunk=4)


# ---------------------------------------------------------------- data ------
def state_repr(qpos, qvel, spec):
    """Observable state (design note 3): the positions that neither wind nor
    are dropped, (cos, sin) of each winding hinge, then every velocity."""
    nq = qpos.shape[-1]
    lin = [i for i in range(nq) if i not in spec["wind"] and i not in spec["drop"]]
    parts = [qpos[..., lin]]
    for i in spec["wind"]:
        parts += [np.cos(qpos[..., i:i + 1]), np.sin(qpos[..., i:i + 1])]
    parts.append(qvel)
    return np.concatenate(parts, -1)


def kin_coeffs(qpos, qvel):
    """Constant-velocity null model for iKCE (arXiv 2607.05966), fit on REAL
    transitions: qpos_i' = qpos_i + c_i qvel_i by least squares, per joint
    (one dof per joint, so qvel_i is qpos_i's velocity)."""
    dp = (qpos[:, 1:] - qpos[:, :-1]).reshape(-1, qpos.shape[-1])
    v = qvel[:, :-1].reshape(-1, qvel.shape[-1])
    return (v * dp).sum(0) / (v * v).sum(0)


class TaskData:
    """One task's dataset: frames through a memory map, the rest in RAM, plus
    the D_train statistics every checkpoint shares."""

    def __init__(self, task):
        self.task, self.spec = task, TASK_STATE[task]
        self.frames = np.load(DATA / task / "frames.npy", mmap_mode="r")
        m = np.load(DATA / task / "meta.npz")
        self.action, self.proprio = m["action"], m["proprio"]
        self.qpos, self.qvel = m["qpos"], m["qvel"]
        self.mjstate_eval = m["mjstate"][N_TRAIN:]
        self.L, self.act_dim = self.action.shape[1], self.action.shape[-1]
        nq = self.qpos.shape[-1]
        self.lin = [i for i in range(nq) if i not in self.spec["wind"] and i not in self.spec["drop"]]
        S = state_repr(self.qpos[:N_TRAIN], self.qvel[:N_TRAIN], self.spec).reshape(-1, len(self.lin) + 2 * len(self.spec["wind"]) + nq)
        self.s_mu, self.s_sd = S.mean(0), S.std(0)
        P = self.proprio[:N_TRAIN].reshape(-1, self.proprio.shape[-1]).astype(np.float64)
        self.p_mu, self.p_sd = P.mean(0), P.std(0)
        self.n_prop, self.n_state = P.shape[1], S.shape[1]
        self.n_pos = len(self.lin) + 2 * len(self.spec["wind"])     # state columns [0, n_pos) are positions
        self.kin_c = kin_coeffs(self.qpos[:N_TRAIN], self.qvel[:N_TRAIN])

    def state(self, qpos, qvel):
        """z-scored observable state."""
        return (state_repr(np.asarray(qpos), np.asarray(qvel), self.spec) - self.s_mu) / self.s_sd

    def targets(self, e, a, b):
        """z-scored probe targets concat(proprio, state) for frames a..b-1 of episode e."""
        p = (self.proprio[e, a:b] - self.p_mu) / self.p_sd
        return np.concatenate([p, self.state(self.qpos[e, a:b], self.qvel[e, a:b])], -1)


_TD = {}


def task_data(task):
    if task not in _TD:
        _TD[task] = TaskData(task)
    return _TD[task]


def fixed_segments(td, S, seed=7):
    """S_M: fixed D_eval segments, identical for every checkpoint, each with
    S.ctx burn-in frames before t = 0."""
    rng = np.random.default_rng(seed)
    cand = [(e, s) for e in range(N_TRAIN, N_EP) for s in range(S.ctx, td.L - S.seg_T + 1)]
    pick = [cand[k] for k in rng.choice(len(cand), S.n_seg, replace=False)]
    fr = np.stack([td.frames[e, s - S.ctx:s + S.seg_T + 1] for e, s in pick])
    ac = np.stack([td.action[e, s - S.ctx:s + S.seg_T] for e, s in pick])
    tg = np.stack([td.targets(e, s, s + S.seg_T + 1) for e, s in pick])
    seg = SegmentBatch(obs=fr[:, S.ctx:], actions=ac[:, S.ctx:], states=tg[..., td.n_prop:])
    return seg, fr[:, :S.ctx], ac[:, :S.ctx], tg


def fixed_roots(n, seed=3, min_t=20, max_t=480):
    """Root states for the interventional audits: (episode, decision) pairs in
    D_eval, restored exactly from the stored MuJoCo state.  max_t must leave a
    full planning horizon of logged actions after the root (callers pass
    L - h_plan); the original 480 assumed h_plan = 16."""
    rng = np.random.default_rng(seed)
    cand = [(e, t) for e in range(N_TRAIN, N_EP) for t in range(min_t, max_t)]
    return [cand[k] for k in rng.choice(len(cand), n, replace=False)]


def pipeline_check(td, n=4, seed=5):
    """Re-render stored D_eval states through the evaluation pipeline and compare
    with frames.npy (design note 1).  Needs a GPU node."""
    env = DMCEnv(td.task, img=128); env.reset(0)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        e, t = int(rng.integers(N_TRAIN, N_EP)), int(rng.integers(0, td.L + 1))
        px = pool(env.set_state(td.mjstate_eval[e - N_TRAIN, t])["pixels"]).astype(int)
        diffs.append(np.abs(px - td.frames[e, t].astype(int)))
    d = np.stack(diffs)
    out = {"render_check_max_diff": int(d.max()), "render_check_frac_diff": float((d > 0).mean())}
    if out["render_check_max_diff"] > 16 or out["render_check_frac_diff"] > 0.05:
        raise RuntimeError(f"rendered frames do not match the training pipeline: {out}")
    return out


# ------------------------------------------------------- model wrappers -----
class BurnIn:
    """The adapter as the metric library sees it on the fixed segments: the
    encoded truth of segment i is the posterior that has first filtered its
    burn-in context (design note 2).  Everything else is delegated.
    The metric library re-encodes the segments in every call; the filter is
    causal and deterministic, so the full encoding is computed once and any
    request for a leading block of the segments' own arrays (all rows up to n,
    all frames up to t) is served from it.  Other inputs are encoded afresh."""

    def __init__(self, ad, ctx_obs, ctx_act, seg, recurrent, chunk):
        self.ad, self.co, self.ca, self.obs, self.act = ad, ctx_obs, ctx_act, seg.obs, seg.actions
        self.rec, self.chunk, self._full = recurrent, chunk, None

    def __getattr__(self, k):
        return getattr(self.ad, k)

    def _leading_block(self, obs, actions):
        at = lambda x: (x.__array_interface__["data"][0], x.strides)
        return (isinstance(obs, np.ndarray) and isinstance(actions, np.ndarray)
                and at(obs) == at(self.obs) and at(actions) == at(self.act)
                and obs.shape[2:] == self.obs.shape[2:] and obs.shape[0] <= self.obs.shape[0]
                and obs.shape[1] <= self.obs.shape[1]
                and actions.shape[:2] == (obs.shape[0], obs.shape[1] - 1))

    def _full_encoding(self):
        if self._full is None:
            self._full = self._encode(self.obs, self.act)
        return self._full

    def encode_sequence(self, obs, actions):
        if self._leading_block(obs, actions):
            return self._full_encoding()[:obs.shape[0], :obs.shape[1]].copy()
        return self._encode(obs, actions)

    def _encode(self, obs, actions):
        out, c = [], self.co.shape[1]
        for i in range(0, len(obs), self.chunk):
            o, a = np.asarray(obs[i:i + self.chunk]), np.asarray(actions[i:i + self.chunk])
            if self.rec:
                o = np.concatenate([self.co[i:i + len(o)], o], 1)
                a = np.concatenate([self.ca[i:i + len(o)], a], 1)
                out.append(self.ad.encode_sequence(o, a)[:, c:])
            else:
                out.append(self.ad.encode_sequence(o, a))
        return np.concatenate(out)

    def encode(self, obs):
        """Filtered latent at the first frame of segments 0..n-1 (used by M9)."""
        obs = np.asarray(obs)
        if not self.rec:
            return self.ad.encode(obs)
        if not np.array_equal(obs, self.obs[:len(obs), 0]):
            raise ValueError("BurnIn.encode only encodes the segments' own first frames")
        return self._full_encoding()[:len(obs), 0].copy()


class Probe:
    """Ridge probe latent -> z-scored concat(proprio, state), fit on filtered
    latents of REAL D_train frames (plan Appendix B).  `predict` (what the
    adapter's probe() calls) returns the z-scored state columns; `proprio`
    returns the proprio columns in raw units, for the U2 policies.  coef_ and
    intercept_ are the state rows, which the JEPA adapter's torch_fns() reads."""

    def __init__(self, ridge, td):
        W, b, P = ridge.coef_, ridge.intercept_, td.n_prop
        self.alpha = float(ridge.alpha_)
        self.Wp, self.bp, self.coef_, self.intercept_ = W[:P], b[:P], W[P:], b[P:]
        self.p_mu, self.p_sd = td.p_mu, td.p_sd

    def predict(self, z):
        return np.asarray(z) @ self.coef_.T + self.intercept_

    def predict_all(self, z):
        z = np.asarray(z)
        return np.concatenate([z @ self.Wp.T + self.bp, self.predict(z)], -1)

    def proprio(self, z):
        return (np.asarray(z) @ self.Wp.T + self.bp) * self.p_sd + self.p_mu


def fit_probe(ad, td, S, seed=0):
    """Windows of ctx + probe_T decisions from D_train; latents after the burn-in.
    One ridge penalty for all (z-scored) targets, chosen by 5-fold CV with whole
    windows held out (design note 9)."""
    rng = np.random.default_rng(seed)
    W = S.ctx + S.probe_T
    eps, st = rng.choice(N_TRAIN, S.probe_win), rng.integers(0, td.L - W + 1, S.probe_win)
    X, Y, G = [], [], []
    for i in range(0, S.probe_win, S.chunk):
        pairs = list(zip(eps[i:i + S.chunk], st[i:i + S.chunk]))
        fr = np.stack([td.frames[e, s:s + W + 1] for e, s in pairs])
        ac = np.stack([td.action[e, s:s + W] for e, s in pairs])
        z = ad.encode_sequence(fr, ac)[:, S.ctx:]
        X.append(z.reshape(-1, z.shape[-1]))
        Y.append(np.concatenate([td.targets(e, s + S.ctx, s + W + 1) for e, s in pairs]))
        G.append(np.repeat(np.arange(i, i + len(pairs)), S.probe_T + 1))
    X, Y, G = np.concatenate(X), np.concatenate(Y), np.concatenate(G)
    folds = list(GroupKFold(min(5, S.probe_win)).split(X, Y, G))
    return Probe(RidgeCV(alphas=ALPHAS, cv=folds).fit(X, Y), td)


def group_r2(pred, true):
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean(0)) ** 2).sum())


def as_torch_input(ad):
    """Let the adapter accept torch tensors already on its device, so U1 can pass
    100 x 512 candidate sequences to get_cost without a round trip through host
    memory.  Numpy inputs take the adapter's own conversion, unchanged."""
    orig = ad._t

    def _t(x):
        if torch.is_tensor(x):
            return x.to(device=ad.dev, dtype=torch.float32)
        return orig(x)
    ad._t = _t
    return ad


# ------------------------------------------------------------- metrics ------
def ikce(x_std, td):
    """M12: mean over segments and steps of ||x_{t+1} - kin(x_t)||^2, in physical
    units, with kin the constant-velocity model (design note 5).  x_std is a
    z-scored state trajectory (N, T+1, n_state)."""
    x = np.asarray(x_std) * td.s_sd + td.s_mu
    nL, nW = len(td.lin), len(td.spec["wind"])
    p, cs, v = x[..., :nL], x[..., nL:nL + 2 * nW], x[..., nL + 2 * nW:]
    res = [p[:, 1:] - p[:, :-1] - td.kin_c[td.lin] * v[:, :-1][..., td.lin]]
    for k, i in enumerate(td.spec["wind"]):
        th = np.arctan2(cs[..., 2 * k + 1], cs[..., 2 * k])
        d = th[:, 1:] - th[:, :-1] - td.kin_c[i] * v[:, :-1, i]
        res.append(((d + np.pi) % (2 * np.pi) - np.pi)[..., None])
    res.append(v[:, 1:] - v[:, :-1])
    return float((np.concatenate(res, -1) ** 2).sum(-1).mean())


# iKCE as the source defines it (arXiv 2607.05966): the constant-velocity null
# model on the ROOT body's position and velocity only.  qpos indices per task:
# cartpole's cart slider, walker's torso height (rootz), cup-catch's free ball.
ROOT_Q = {"cartpole-swingup": (0,), "walker-walk": (0,), "cup-catch": (2, 3)}


def ikce_root(x_std, td, task):
    x = np.asarray(x_std) * td.s_sd + td.s_mu
    v = x[..., td.n_pos:]
    res = []
    for q in ROOT_Q[task]:
        c = td.lin.index(q)
        res.append(x[:, 1:, c] - x[:, :-1, c] - td.kin_c[q] * v[:, :-1, q])
        res.append(v[:, 1:, q] - v[:, :-1, q])
    return float((np.stack(res, -1) ** 2).sum(-1).mean())


def observable_fraction(ad, z_states, C, act_dim, K, tol=1e-3):
    """M10: reward observability fraction (arXiv 2607.01736, eq. 7) and the
    observable / controllable rank fractions that make up CROF.  The LunarLander
    computation (experiments/lunar/evaluate.py) with C = the probe's map onto
    the state columns (design note 4); autograd Jacobians at D_eval latents,
    over a K-step horizon."""
    dev = ad.dev
    f, _, rewf = ad.torch_fns()
    C = np.asarray(C, np.float64)
    rofs, kos, kcs = [], [], []
    for zs in z_states:
        x = torch.as_tensor(zs, dtype=torch.float32, device=dev)
        a0 = torch.zeros(act_dim, device=dev)
        A = torch.autograd.functional.jacobian(lambda x_: f(x_, a0), x)
        Bm = torch.autograd.functional.jacobian(lambda a_: f(x, a_), a0)
        r = torch.autograd.functional.jacobian(rewf, x)
        A, Bm, r = (t.double().cpu().numpy() for t in (A, Bm, r))
        n = A.shape[0]
        # observable subspace: rows of C, C A, ..., C A^K, orthonormalised as we go
        V = np.linalg.qr(C.T)[0].T; blk = C
        for _ in range(K):
            blk = blk @ A
            blk = blk / max(np.linalg.norm(blk), 1e-12)
            U, sv, Vt = np.linalg.svd(np.vstack([V, blk]), full_matrices=False)
            V = Vt[sv > tol * sv[0]]
        rofs.append(float(np.sum((V @ r) ** 2) / max(r @ r, 1e-12)))
        kos.append(V.shape[0] / n)
        # controllable subspace: columns of B, A B, ..., A^K B
        W = np.linalg.qr(Bm)[0]; blk = Bm
        for _ in range(K):
            blk = A @ blk
            blk = blk / max(np.linalg.norm(blk), 1e-12)
            U, sv, Vt = np.linalg.svd(np.hstack([W, blk]), full_matrices=False)
            W = U[:, sv > tol * sv[0]]
        kcs.append(W.shape[1] / n)
    return float(np.mean(rofs)), float(np.mean(kos)), float(np.mean(kcs))


def replay(env, root, actions):
    """Restore a logged MuJoCo state exactly and apply an action sequence.
    dm_control's step counter is reset first (design note 6)."""
    env.set_state(root)
    env.env._step_count, env.env._reset_next_step = 0, False
    ret, o = 0.0, None
    for a in actions:
        o, r, done = env.step(a); ret += r
        if done:
            raise RuntimeError("replay reached the time limit")
    return o, ret


def root_latent(ad, td, e, t, recurrent):
    """Latent at decision t of D_eval episode e: the posterior filtered over the
    whole observed prefix (RSSM), or the encoding of frame t (JEPA)."""
    if recurrent:
        return ad.encode_sequence(np.asarray(td.frames[e, :t + 1])[None], td.action[e, :t][None])[:, -1]
    return ad.encode(np.asarray(td.frames[e, t])[None])


def interventional_audits(ad, td, roots, rng, S, recurrent):
    """M11 plan-cost discrepancy (arXiv 2607.10362) and M7 matched-intervention
    audit (arXiv 2608.29998), both exact via set_state replay.  True cost is the
    negated sum of true rewards over H_plan decisions (no rendering)."""
    env = DMCEnv(td.task, render=False); env.reset(0)
    H, A = S.h_plan, td.act_dim
    g_sup, g_mean, rhos, dsim, dwm = [], [], [], [], []
    for (e, t) in roots:
        root = td.mjstate_eval[e - N_TRAIN, t]
        z = root_latent(ad, td, e, t, recurrent)
        logged = td.action[e, t:t + H]
        U = np.concatenate([
            np.clip(logged[None] + s * rng.standard_normal((16, H, A)), -1, 1)
            for s in (0.1, 0.4)] + [rng.uniform(-1, 1, (32, H, A))]).astype(np.float32)
        J = np.array([-replay(env, root, u)[1] for u in U])
        Jh = ad.get_cost(np.repeat(z, len(U), 0), U)
        d = np.abs(Jh - J); g_sup.append(d.max()); g_mean.append(d.mean())
        rhos.append(spearmanr(Jh, J).statistic if np.std(J) > 0 and np.std(Jh) > 0 else np.nan)
        # M7: pairs differing only in the first action, 5-step effect on the state
        for _ in range(5):
            u1 = rng.uniform(-1, 1, (5, A)).astype(np.float32); u2 = u1.copy()
            u2[0] = rng.uniform(-1, 1, A)
            o1, _ = replay(env, root, u1); o2, _ = replay(env, root, u2)
            zz = ad.rollout(np.repeat(z, 2, 0), np.stack([u1, u2]))[:, -1]
            p = ad.probe(zz)
            dsim.append(td.state(o1["qpos"], o1["qvel"]) - td.state(o2["qpos"], o2["qvel"]))
            dwm.append(p[0] - p[1])
    dsim, dwm = np.asarray(dsim), np.asarray(dwm)
    sd = dsim.std(0) + 1e-9; dsim, dwm = dsim / sd, dwm / sd        # per-dim scale
    cross = float(np.mean((dwm * dsim).sum(-1)))
    vw, vs = float(np.mean((dwm ** 2).sum(-1))), float(np.mean((dsim ** 2).sum(-1)))
    rho = np.asarray(rhos, float)
    return {"M11_g_sup": float(np.mean(g_sup)), "M11_mean_abs": float(np.mean(g_mean)),
            "M11_one_minus_rho": float(1 - np.nanmean(rho)) if np.isfinite(rho).any() else float("nan"),
            "M11_n_rho": int(np.isfinite(rho).sum()),
            "M7_gain": cross / vs, "M7_r2": cross ** 2 / (vw * vs),
            "M7_cosine": cross / np.sqrt(vw * vs)}


# ----------------------------------------------------------- utilities ------
def mppi_step(ad, Z, ws, cfg, noise, amp, dev):
    """One MPPI decision for E episodes at once: the update of planners/mppi.py
    MPPI.plan, batched over episodes (design note 7).  Each iteration is ONE
    get_cost call over E x population sequences.  Returns the first action of
    the final mean, the warm start for the next decision, and the number of
    non-finite costs."""
    E, H, A = ws.shape
    P = cfg.population
    Zr = torch.as_tensor(np.asarray(Z), dtype=torch.float32, device=dev).repeat_interleave(P, 0)
    mu = ws.clone()
    std = torch.full_like(mu, cfg.init_std)
    bad = 0
    for _ in range(cfg.iterations):
        seq = (mu[:, None] + std[:, None] * noise((E, P, H, A))).clamp(cfg.action_low, cfg.action_high)
        with torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext():
            J = ad.get_cost(Zr, seq.reshape(E * P, H, A).float())
        J = torch.as_tensor(J, dtype=torch.float64, device=dev).reshape(E, P)
        nf = ~torch.isfinite(J)
        bad += int(nf.sum())
        J = torch.where(nf, torch.full_like(J, 1e9), J)
        w = torch.exp(-(J - J.min(1, keepdim=True).values) / max(cfg.temperature, 1e-8))
        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        new_mu = torch.einsum("ep,epha->eha", w, seq)
        new_std = torch.sqrt(torch.einsum("ep,epha->eha", w, (seq - new_mu[:, None]) ** 2))
        mu = (1 - cfg.momentum) * new_mu + cfg.momentum * mu
        std = new_std.clamp_min(cfg.min_std)
    nxt = torch.roll(mu, -1, 1)
    nxt[:, -1] = 0.0
    if not cfg.warm_start:
        nxt.zero_()
    return mu[:, 0].float().cpu().numpy(), nxt, bad


def plan_config(S):
    return PlanConfig(horizon=S.h_plan, receding_horizon=1, population=S.pop, iterations=S.iters,
                      action_low=-1.0, action_high=1.0)


def frame(o, S):
    return np.zeros((64, 64, 3), np.uint8) if S.fake_render else pool(o["pixels"])


def _env_worker(conn, task, seeds, fake):
    """A slice of the U1 episodes in its own process, with its own EGL context."""
    S = Settings(fake_render=fake)
    envs = [DMCEnv(task, img=128, render=not fake) for _ in seeds]
    conn.send(np.stack([frame(e.reset(s), S) for e, s in zip(envs, seeds)]))
    while True:
        acts = conn.recv()
        if acts is None:
            return
        out = [e.step(a) for e, a in zip(envs, acts)]
        conn.send((np.stack([frame(o, S) for o, _, _ in out]),
                   np.array([r for _, r, _ in out]), np.array([d for _, _, d in out])))


class EnvPool:
    """The U1 episodes in lockstep: physics, 128x128 render and pooling for every
    episode, in this process (workers = 0) or split over worker processes.
    The layout changes wall time only; frames and rewards are the same."""

    def __init__(self, task, seeds, S, workers=0):
        self.S, self.workers = S, workers
        if workers:
            ctx = mp.get_context("spawn")
            self.parts = np.array_split(np.arange(len(seeds)), workers)
            self.conns, self.procs = [], []
            for idx in self.parts:
                a, b = ctx.Pipe()
                pr = ctx.Process(target=_env_worker, args=(b, task, [seeds[i] for i in idx], S.fake_render),
                                 daemon=True)
                pr.start(); self.conns.append(a); self.procs.append(pr)
            self.frames0 = np.concatenate([c.recv() for c in self.conns])
        else:
            self.envs = [DMCEnv(task, img=128, render=not S.fake_render) for _ in seeds]
            self.frames0 = np.stack([frame(e.reset(s), S) for e, s in zip(self.envs, seeds)])

    def step(self, act):
        if self.workers:
            for c, idx in zip(self.conns, self.parts):
                c.send(act[idx])
            res = [c.recv() for c in self.conns]
            return tuple(np.concatenate(x) for x in zip(*res))
        out = [e.step(a) for e, a in zip(self.envs, act)]
        return (np.stack([frame(o, self.S) for o, _, _ in out]),
                np.array([r for _, r, _ in out]), np.array([d for _, _, d in out]))

    def close(self):
        if self.workers:
            for c in self.conns:
                c.send(None)
            for pr in self.procs:
                pr.join(timeout=30)


def u1_planning(ad, task, S, dev, amp, seed=777, workers=0):
    """Batched MPPI episodes in lockstep; each episode's latent is filtered with
    its own history (filter_step on the pooled rendered frame), never
    re-encoded from a single frame."""
    cfg = plan_config(S)
    seeds = U1_SEEDS[:S.u1_episodes]
    E = len(seeds)
    A = task_data(task).act_dim
    envs = EnvPool(task, seeds, S, workers)
    try:
        Z = ad.encode(envs.frames0)
        gen = torch.Generator(device=dev).manual_seed(seed)
        noise = lambda shape: torch.randn(shape, generator=gen, device=dev).double()
        ws = torch.zeros((E, cfg.horizon, A), dtype=torch.float64, device=dev)
        ret, bad, tp, te, T = np.zeros(E), 0, 0.0, 0.0, 0
        for T in range(1, S.u1_T + 1):
            t0 = time.time()
            act, ws, b = mppi_step(ad, Z, ws, cfg, noise, amp, dev); bad += b
            t1 = time.time()
            frames, r, done = envs.step(act)
            ret += r
            te += time.time() - t1; tp += t1 - t0
            Z = ad.filter_step(Z, act, frames)
            if done.all():
                break
    finally:
        envs.close()
    return ret, {"U1_decisions": T, "U1_nonfinite_costs": bad, "t_u1_plan_s": tp, "t_u1_env_s": te}


def u2_imagined_returns(ad, probe, task, S, dev, cache):
    """Each policy acts on the probe-decoded proprio inside the world model,
    from the encoded first frame of each fixed start state."""
    seeds = U2P.U2_SEEDS[:S.u2_seeds]
    start = U2P.start_frames(task, seeds, cache=cache, fake=S.fake_render)
    prior = U2P.load_prior(task, dev)
    dr = U2P.crn_draws(prior.act_dim, seeds, S.u2_H)
    n_pol = len(U2P.POLICIES)
    Z = np.tile(ad.encode(start), (n_pol, 1))                # policy-major rows
    R, alive = np.zeros(len(Z)), np.ones(len(Z))
    on_action = getattr(ad, "reward_needs_action", False)      # TD-MPC2: r(z_t, a_t), design note 11
    for t in range(S.u2_H):
        a = U2P.policy_actions(prior.act(probe.proprio(Z).astype(np.float32)), dr, t)
        if on_action:
            r = ad.reward(Z, a); Z = ad.step(Z, a)
        else:
            Z = ad.step(Z, a); r = ad.reward(Z)
        R += alive * r
        alive = alive * ad.cont_prob(Z)
    return R.reshape(n_pol, -1).mean(1)


# ---------------------------------------------------------------- main ------
_CHECKED = {}


def evaluate(ckpt_path, dev, S=Settings(), parts=("metrics", "u1", "u2"), env_workers=0):
    t0 = time.time()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    family, task = ck["family"], ck["task"]
    hist = family == "jepa" and ck.get("arch") == "hist3"          # plan amendment A11
    if family == "tdmpc2":
        model = PixelTDMPC2(act_dim=ck["act_dim"], size=ck["size"])
        model.load_state_dict(ck["state_dict"])
        ad = as_torch_input(TDMPC2Adapter(model, dev))
    else:
        model = (PixelJEPAHist if hist else {"rssm": PixelRSSM, "jepa": PixelJEPA}[family])(
            act_dim=ck["act_dim"], width=ck["width"])
        model.load_state_dict(ck["state_dict"])
        ad = as_torch_input((PixelJEPAHistAdapter if hist else
                             {"rssm": PixelRSSMAdapter, "jepa": PixelJEPAAdapter}[family])(model, dev))
    rec = family in ("rssm", "tdmpc2") or hist   # history-dependent latent: burn-in and prefix roots
    td = task_data(task)
    out = {k: ck[k] for k in ("family", "task", "run", "seed", "width", "data_frac", "step", "frac", "val_loss")}
    out["arch"] = ck.get("arch", "single")   # jepa: hist3 = 3-frame history (A11); tdmpc2: tdmpc2_<size>M
    out["size"] = ck.get("size")
    out["ckpt"] = str(ckpt_path)
    out["gpu"] = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    out["eval_settings"] = dataclasses.asdict(S)
    if not S.fake_render:
        if task not in _CHECKED:
            _CHECKED[task] = pipeline_check(td)
        out.update(_CHECKED[task])
    probe = fit_probe(ad, td, S)
    ad.attach_probe(probe)
    out["probe_alpha"] = probe.alpha
    seg, cobs, cact, tg = fixed_segments(td, S)
    adc = BurnIn(ad, cobs, cact, seg, rec, S.chunk)
    N, T, D, nS = seg.n, seg.horizon, ad.latent_dim, td.n_state
    # held-out probe R^2 per group, on the D_eval segments' encoded truth
    z_true = adc.encode_sequence(seg.obs, seg.actions).reshape(-1, D)
    ph, tt = probe.predict_all(z_true), tg.reshape(-1, tg.shape[-1])
    P, Q = td.n_prop, td.n_prop + td.n_pos
    out["probe_r2_heldout_proprio"] = group_r2(ph[:, :P], tt[:, :P])
    out["probe_r2_heldout_qpos"] = group_r2(ph[:, P:Q], tt[:, P:Q])
    out["probe_r2_heldout_qvel"] = group_r2(ph[:, Q:], tt[:, Q:])
    out["probe_r2_heldout_state"] = group_r2(ph[:, P:], tt[:, P:])
    del z_true, ph, tt
    if "metrics" in parts:
        # M4 / M5 under the three protocols, three measurement spaces
        for pname, R in (("P1", 1), ("P3", S.h_plan), ("P2", INF)):
            for space in ("raw", "normalised", "probe"):
                out[f"latent_err_{pname}_{space}"] = float(
                    latent_error(adc, seg, lag=1, resync_every=R, space=space).mean())
            flat = predict_with_resync(adc, seg, R)[0][:, 1:].reshape(-1, D)
            if ad.has_decoder:
                target = seg.obs[:, 1:].reshape(-1, 64, 64, 3)
                sse = sum(float(((ad.decode(flat[i:i + 1024]) - (target[i:i + 1024] / 255.0 - 0.5)) ** 2).sum())
                          for i in range(0, len(flat), 1024))
                out[f"obs_rmse_{pname}"] = float(np.sqrt(sse / target.size))
            else:
                out[f"obs_rmse_{pname}"] = float("nan")      # no decoder: not applicable
            prb = ad.probe(flat).reshape(N, T, nS)
            out[f"probe_rmse_{pname}"] = float(np.sqrt(((prb - seg.states[:, 1:]) ** 2).mean()))
        for k in (1, S.h_plan // 2, S.h_plan):          # one step, half and full planning horizon
            out[f"M8_probe_r2_k{k}"] = float(probe_accuracy(adc, seg, k, resync_every=INF))
        small = SegmentBatch(seg.obs[:S.m9_n], seg.actions[:S.m9_n], seg.states[:S.m9_n])
        out[f"M9_sample_disagreement_k{S.h_plan}"] = (
            sample_disagreement(adc, small, k=S.h_plan, n_samples=8) if ad.is_stochastic else float("nan"))
        zs = adc.encode_sequence(seg.obs[:40, :11], seg.actions[:40, :10])[:, -1]
        out["M10_ROF"], out["M10_k_obs"], out["M10_k_ctrl"] = observable_fraction(
            ad, zs[:S.m10_n], probe.coef_, td.act_dim, K=S.h_plan)
        out["M10_CROF"] = (out["M10_ROF"] + (1 - out["M10_k_ctrl"]) + (1 - out["M10_k_obs"])
                           + out["probe_rmse_P2"])
        z0 = adc.encode_sequence(seg.obs[:, :1], seg.actions[:, :0])[:, 0]
        imag = ad.probe(ad.rollout(z0, seg.actions[:, :S.h_plan]).reshape(-1, D)).reshape(N, S.h_plan + 1, nS)
        out["M12_ikce_model"], out["M12_ikce_real"] = ikce(imag, td), ikce(seg.states[:, :S.h_plan + 1], td)
        out["M12_ikce_log_ratio"] = float(abs(np.log(out["M12_ikce_model"] / out["M12_ikce_real"])))
        out["M12_ikce_root_model"] = ikce_root(imag, td, task)
        out["M12_ikce_root_real"] = ikce_root(seg.states[:, :S.h_plan + 1], td, task)
        out["M12_ikce_root_log_ratio"] = float(abs(np.log(max(out["M12_ikce_root_model"], 1e-12)
                                                          / max(out["M12_ikce_root_real"], 1e-12))))
        rng = np.random.default_rng(12345)
        out.update(interventional_audits(ad, td, fixed_roots(S.n_roots, seed=3, max_t=td.L - S.h_plan),
                                         rng, S, rec))
    out["t_metrics_s"] = time.time() - t0
    u2_default = (S.u2_seeds, S.u2_H) == (Settings.u2_seeds, Settings.u2_H)   # what the U2 caches hold
    if "u1" in parts:
        t1 = time.time()
        amp = dev == "cuda"
        if amp:
            check_precision(autocast_dtype=torch.float16)        # plan amendment A3
        ret, info = u1_planning(ad, task, S, dev, amp, workers=env_workers)
        out["U1_return"] = float(ret.mean()); out["U1_returns"] = ret.tolist()
        out.update(info)
        out["t_u1_s"] = time.time() - t1
    if "u2" in parts:
        t2 = time.time()
        true_r = U2P.true_returns(task, U2P.U2_SEEDS[:S.u2_seeds], S.u2_H, cache=u2_default)[0]
        imag_r = u2_imagined_returns(ad, probe, task, S, dev, cache=u2_default)
        out["U2_kendall_tau"] = float(kendalltau(imag_r, true_r).statistic)
        out["U2_imagined"] = imag_r.tolist(); out["U2_true"] = true_r.tolist()
        out["t_u2_s"] = time.time() - t2
    out["t_total_s"] = time.time() - t0
    return out


def planned_steps(run, family="rssm"):
    n = BUDGET[family]
    if n is None:
        raise ValueError(f"no step budget fixed for {family} yet")
    if run in ("seed0", "seed1", "seed2"):
        return tuple(max(1, int(round(f * n))) for f in POPW_FRACS)
    return (n,)


def planned_checkpoints(families=FAMILIES):
    return [CKPT / f / t / r / f"step{s:07d}.pt" for t in TASKS for f in families if BUDGET[f]
            for r in FAMILY_RUNS[f] for s in planned_steps(r, f)]


def write_progress(family, task, run, status, error=None):
    """results/progress/dmc_eval/<family>_<task>_<run>.json for the dashboard.
    planned_eval is 0 on purpose: the training progress files carry the plan."""
    rd = RESULTS / "dmc" / family / task / run
    steps = planned_steps(run, family)
    n = sum((rd / f"step{s:07d}.json").exists() for s in steps)
    if status == "running" and n >= len(steps):
        status = "done"
    d = {"experiment": "DMControl evaluation", "kind": "eval", "family": family, "task": task,
         "run": run, "unit": "checkpoint", "done": n, "total": len(steps), "status": status,
         "evaluated": n, "planned_eval": 0,
         "updated_iso": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    if error:
        d["error"] = error[:300]
    PROG.mkdir(parents=True, exist_ok=True)
    f = PROG / f"{family}_{task}_{run}.json"
    tmp = f.with_suffix(f".tmp{os.getpid()}"); tmp.write_text(json.dumps(d)); tmp.replace(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--manifest", help="write the planned checkpoint paths to this file and exit")
    ap.add_argument("--families", default=",".join(FAMILIES), help="families for --manifest")
    ap.add_argument("--prepare", nargs="+", metavar="TASK",
                    help="build the per-task caches (U2 true returns, U2 start frames) and run the "
                         "render pipeline check; needs a GPU node")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke-test settings (fake frames)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override Settings fields")
    ap.add_argument("--parts", default="metrics,u1,u2")
    ap.add_argument("--out", help="result directory; required with --tiny/--set/--parts")
    ap.add_argument("--env-workers", type=int, default=int(os.environ.get("DMC_ENV_WORKERS", 0)),
                    help="processes that step and render the U1 episodes (0 = in this process); "
                         "wall time only, results are identical")
    a = ap.parse_args()
    if a.manifest:
        paths = planned_checkpoints(tuple(a.families.split(",")))
        pathlib.Path(a.manifest).write_text("\n".join(map(str, paths)) + "\n")
        print(f"wrote {len(paths)} paths to {a.manifest}"); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    if a.prepare:
        for task in a.prepare:
            t = time.time()
            m, _ = U2P.true_returns(task)
            fr = U2P.start_frames(task)
            print(f"{task}: U2 true {np.round(m, 1).tolist()}, start frames {fr.shape}, "
                  f"{pipeline_check(task_data(task))} ({time.time() - t:.0f}s)", flush=True)
        return
    S = Settings(**TINY) if a.tiny else Settings()
    for kv in a.set:
        k, v = kv.split("=")
        setattr(S, k, type(getattr(S, k))(v) if not isinstance(getattr(S, k), bool) else v == "1")
    parts = tuple(a.parts.split(","))
    official = S == Settings() and set(parts) == {"metrics", "u1", "u2"} and not a.out
    if not official and not a.out:
        ap.error("non-default settings or parts need --out (results/dmc holds full evaluations only)")
    failed, touched = [], set()
    for c in a.ckpts:
        c = pathlib.Path(c)
        family, task, run = c.parts[-4], c.parts[-3], c.parts[-2]
        res_dir = (RESULTS / "dmc" / family / task / run) if official else pathlib.Path(a.out) / family / task / run
        f = res_dir / (c.stem + ".json")
        if f.exists() and not os.environ.get("FORCE"):
            print("skip (done)", f, flush=True); continue
        if not c.exists():
            print("skip (checkpoint not written yet)", c, flush=True); continue
        if official:
            write_progress(family, task, run, "running"); touched.add((family, task, run))
        try:
            r = evaluate(c, dev, S, parts, env_workers=a.env_workers)
        except Exception as e:
            traceback.print_exc()
            failed.append(str(c))
            if official:
                write_progress(family, task, run, "failed", f"{c.name}: {type(e).__name__}: {e}")
                touched.discard((family, task, run))
            continue
        res_dir.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".tmp{os.getpid()}")
        json.dump(r, open(tmp, "w"), indent=1); tmp.replace(f)
        if official:
            write_progress(family, task, run, "running")
        print(f"{family}/{task}/{run}/{c.stem}: U1={r.get('U1_return', float('nan')):.1f} "
              f"U2_tau={r.get('U2_kendall_tau', float('nan')):.3f} val={r['val_loss']:.3f} "
              f"latP1={r.get('latent_err_P1_raw', float('nan')):.4f} "
              f"({r['t_metrics_s']:.0f}s metrics, {r.get('t_u1_s', 0):.0f}s U1, "
              f"{r.get('t_u2_s', 0):.0f}s U2, {r['t_total_s']:.0f}s total)", flush=True)
    for family, task, run in touched:
        steps = planned_steps(run, family)
        n = sum((RESULTS / "dmc" / family / task / run / f"step{s:07d}.json").exists() for s in steps)
        write_progress(family, task, run, "done" if n >= len(steps) else "queued")
    if failed:
        sys.exit(f"{len(failed)} checkpoint(s) failed: {failed}")


if __name__ == "__main__":
    main()
