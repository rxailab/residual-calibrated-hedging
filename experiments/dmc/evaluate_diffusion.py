"""Evaluate one diffusion-family checkpoint (fine-tuned MiniWorld-0.5B, one
model for all three DMControl tasks) on one task: the offline metrics and the
closed-loop utilities of experiments/dmc/evaluate.py, written to storage as
results/dmc/diffusion/<task>/<run>/stepNNNNNNN.json.

Everything is measured in blocks of 4 decisions, the model's time step
(wm/diffusion.py: latent frame k encodes frames 4k-3 .. 4k).  Plan Section 7:
"a family that plans in action blocks rolls out H/block predictor steps".

Design choices, for the pre-registration log (numbering continues evaluate.py's)
D1. Latent, probe, reward head.  The latent is the frozen VAE latent, the same
    function for every checkpoint, so the probe and the reward head are fit once
    per task (experiments/dmc/diffusion_heads.py) and shared; checkpoints differ
    only in their dynamics.  The probe reads the state at frame 4k from latent
    k; the reward head predicts the block reward (sum of 4 rewards).
D2. Segments.  500 fixed D_eval segments of 16 blocks (64 decisions) start at a
    block boundary k0, with the 4 latent frames ending at k0 (16 decisions) as
    the clean context: the burn-in of the other families.  Their encoded truth
    is the stored pre-encoding of D (fp32 VAE, the evaluation's own encoder).
    The same (episode, start) draw as the other families is not possible
    because their segments start anywhere; the draw is fixed for this family.
D3. Protocols in blocks: P1 resynchronises every block, P3 every H_plan = 8
    blocks (32 decisions), P2 never.  Errors are averaged over all 16
    predicted blocks, as latent_error averages over all 64 decisions.  Every
    prediction of block t of segment i uses the same fixed sampler noise under
    every protocol and for every checkpoint (common random numbers).
D4. Horizon-indexed metrics: k = 1 decision has no block equivalent, so M8 is
    reported at k = 4, 16 and 32 decisions (blocks 1, 4, 8); M9 at 32.
D5. Pixel metrics (M3 PSNR and obs_rmse, the RSSM's pixel metric): predicted
    latents are decoded after up to 16 true latents of the episode as decoder
    warm-up (0.0013 RMSE from the episode-aligned decode; 4 latents give 0.015),
    pooled 2x2 to 64x64 and compared with the stored 64x64 frames, in pixel/255
    units like the RSSM's obs_rmse, on the first 200 segments (decoding is the
    cost).  P1 predictions are decoded as one sequence.  FVD and LPIPS (M1, M2)
    are not computed for any family.
D6. M10.  The state of the linearised dynamics is the 4-frame context window
    (as for the stacked-embedding JEPA), the one-block map is the sampler's
    deterministic ODE with fixed noise, and the observable / controllable
    subspaces are built from vector-Jacobian / Jacobian-vector products
    instead of full 12288 x 12288 Jacobians.
D7. M7, M11 roots are D_eval decisions at block boundaries; M7 contrasts
    2-block (8-decision) sequences differing in their first action.
D8. M12 uses the constant-velocity model over one block, qpos_{t+4} =
    qpos_t + c4 qvel_t (diffusion_heads.py).
D9. U1.  MPPI over 4-decision blocks: horizon 8 blocks (32 decisions), the
    common update rule (evaluate.mppi_step, batched over episodes), warm start
    shifted by the executed blocks, the sampler with fresh noise.  Episodes
    are encoded online with the streaming VAE encoder.  Population,
    iterations, replanning interval, sampler steps and K come from Settings,
    set by the G1 timing gate and the pilot-checkpoint tuning (plan App. A).
D10. U2.  Imagined rollouts run 25 blocks (100 decisions) from the encoded
    first frame; within a block the policy acts on the probe state at the
    block's start (the model has no latent in between), with each decision's
    own common-random-number draw.
D11. A candidate plan whose predicted cost is not finite (fp16 overflow of an
    out-of-domain prediction; seen with the pretrained model) gets MPPI weight
    zero, all candidates equal weight if none is finite; U1_nonfinite_frac
    records how often.  Finite predictions plan exactly as before.
D12. Extra U1 episodes (amendment A20) run in chunks: u1_first selects the next
    seeds of the fixed U1 list, and each chunk seeds its own planner and sampler
    noise (777 + u1_first, 1000 + u1_first), so chunks are independent of each
    other and of the first 15 episodes; scripts/diffusion_u1_merge.py joins them.
"""
import os, sys, pathlib, json, time, argparse, datetime, traceback, dataclasses
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from scipy.stats import spearmanr, kendalltau
import wm.diffusion as WD
from experiments.dmc import evaluate as EV
from experiments.dmc import u2_policies as U2P
from experiments.dmc.u2_policies import pool
from experiments.dmc.train_diffusion import cond_table, LAT, N_LAT
from experiments.dmc.diffusion_heads import RewardHead, block_rewards
from envs.dmc import DMCEnv
from planners.mppi import PlanConfig
from wm.rssm import symexp
from paths import RESULTS

BLOCK = WD.BLOCK
HEADS = RESULTS / "diffusion" / "heads"
PROG = RESULTS / "progress" / "dmc_eval"
INF = EV.INF


@dataclasses.dataclass
class Settings:
    n_seg: int = 500
    seg_blocks: int = 16      # 64 decisions
    ctx: int = 4              # context latent frames (16 decisions)
    h_plan: int = 8           # blocks (32 decisions, amendment A10)
    n_pix: int = 200          # segments for the pixel metrics (D5)
    n_roots: int = 60
    m9_n: int = 200
    m10_n: int = 12
    m7_blocks: int = 2
    steps: int = 2            # tuned (DIFFUSION_U1_TUNING.md), for metrics and utilities alike
    ardiff: int = 2
    cfg_scale: float = 1.0
    micro_batch: int = 64     # 128 ran out of memory on a 32 GB V100 in the 16-block P2 rollouts
    u1_episodes: int = 15     # plan G1 fallback (V100 closed loop > 25 min per episode)
    u1_first: int = 0         # index of the first U1 seed: extra episodes run in chunks (D12)
    u1_T: int = 500
    pop: int = 64
    iters: int = 2
    replan_blocks: int = 2    # 8 decisions, tuned (DIFFUSION_U1_TUNING.md; plan App. A)
    u2_seeds: int = 50
    u2_H: int = 100


TINY = dict(n_seg=6, n_pix=6, seg_blocks=4, h_plan=2, n_roots=2, m9_n=4, m10_n=1, steps=2, ardiff=1,
            u1_episodes=2, u1_T=16, pop=4, iters=1, u2_seeds=3, u2_H=8)


# ------------------------------------------------------------------- data ------
class HeadProbe:
    """The shared ridge probe (D1), with evaluate.Probe's interface."""

    def __init__(self, h, td):
        W, b, P = h["probe_coef"], h["probe_intercept"], h["n_prop"]
        self.alpha = h["probe_alpha"]
        self.Wp, self.bp, self.coef_, self.intercept_ = W[:P], b[:P], W[P:], b[P:]
        self.p_mu, self.p_sd = td.p_mu, td.p_sd

    def _flat(self, z):
        z = np.asarray(z, np.float64)
        return z.reshape(*z.shape[:-3], -1) if z.ndim >= 4 else z

    def predict(self, z):
        return self._flat(z) @ self.coef_.T + self.intercept_

    def proprio(self, z):
        return (self._flat(z) @ self.Wp.T + self.bp) * self.p_sd + self.p_mu


class DiffTask:
    def __init__(self, task, dev):
        self.task, self.ti = task, WD.TASKS.index(task)
        self.td = EV.task_data(task)
        self.lat = np.concatenate([np.load(LAT / task / f"lat_{k:03d}.npy") for k in (38, 39)]).astype(np.float32)
        m = np.load(EV.DATA / task / "meta.npz")
        self.cond = cond_table(m["action"][EV.N_TRAIN:].astype(np.float32), self.ti)    # (100, 126, C)
        self.brew = block_rewards(m["reward"][EV.N_TRAIN:].astype(np.float64))            # (100, 125)
        h = torch.load(HEADS / f"{task}.pt", weights_only=False)
        self.heads = h
        self.probe = HeadProbe(h, self.td)
        self.rhead = RewardHead().to(dev).eval()
        self.rhead.load_state_dict(h["reward_state_dict"])
        self.td4 = _with_kin(self.td, h["kin4"])
        self.dev = dev

    @torch.no_grad()
    def block_reward(self, z):
        """(..., 48, 8, 8) latents -> predicted block rewards (numpy or torch in, torch out)."""
        z = torch.as_tensor(z, dtype=torch.float32, device=self.dev)
        return symexp(self.rhead(z))

    def states(self, e, k0, n):
        """z-scored state at frames 4k0, 4(k0+1), ... (n blocks) of D_eval episode e (0..99)."""
        x = self.td.targets(EV.N_TRAIN + e, 4 * k0, 4 * (k0 + n - 1) + 1)[::BLOCK]
        return x[:, self.td.n_prop:]


def _with_kin(td, kin4):
    """A view of TaskData whose constant-velocity coefficients are the one-block ones (D8)."""
    class V:
        pass
    v = V(); v.__dict__.update(td.__dict__); v.kin_c = np.asarray(kin4)
    return v


def block_segments(dt, S, seed=7):
    """D2: (episode, k0) pairs, true latents, contexts, conditions, states, true 64x64 frames."""
    rng = np.random.default_rng(seed)
    cand = [(e, k) for e in range(100) for k in range(S.ctx - 1, N_LAT - S.seg_blocks)]
    pick = [cand[i] for i in rng.choice(len(cand), S.n_seg, replace=False)]
    T, C = S.seg_blocks, S.ctx
    full = np.stack([dt.lat[e, k - C + 1:k + T + 1] for e, k in pick])                # (N, C-1+T+1, ...)
    cond = np.stack([dt.cond[e, k - C + 1:k + T + 1] for e, k in pick])
    states = np.stack([dt.states(e, k, T + 1) for e, k in pick])
    frames = np.stack([dt.td.frames[EV.N_TRAIN + e, 4 * k + 1:4 * (k + T) + 1] for e, k in pick])  # (N, 4T, 64, 64, 3)
    return {"pick": pick, "full": full, "cond": cond, "states": states, "frames": frames, "C": C, "T": T}


# ------------------------------------------------------------- prediction ------
def predict_resync(samp, seg, R, noise):
    """D3: predicted latents (N, T+1, 48, 8, 8) under resynchronisation every R
    blocks; [:, 0] is the truth.  noise (N, T, 48, 8, 8) is fixed per block."""
    full, cond, C, T = seg["full"], seg["cond"], seg["C"], seg["T"]
    N = len(full)
    pred = np.empty((N, T + 1) + full.shape[2:], np.float32)
    pred[:, 0] = full[:, C - 1]
    if R == 1:
        idx = [(i, t) for t in range(T) for i in range(N)]
        ctx = np.stack([full[i, t:t + C] for i, t in idx]); cc = np.stack([cond[i, t:t + C] for i, t in idx])
        fc = np.stack([cond[i, C + t:C + t + 1] for i, t in idx]); nz = np.stack([noise[i, t:t + 1] for i, t in idx])
        out = samp(ctx, torch.as_tensor(cc), torch.as_tensor(fc), torch.as_tensor(nz)).cpu().numpy()
        for j, (i, t) in enumerate(idx):
            pred[i, t + 1] = out[j, 0]
        return pred
    step = T if R >= INF else R
    for s in range(0, T, step):
        F_ = min(step, T - s)
        out = samp(full[:, s:s + C], torch.as_tensor(cond[:, s:s + C]), torch.as_tensor(cond[:, C + s:C + s + F_]),
                   torch.as_tensor(noise[:, s:s + F_])).cpu().numpy()
        pred[:, s + 1:s + F_ + 1] = out
        if s + F_ < T:
            pass                                    # the next stretch restarts from the truth
    return pred


def pixel_errors(vae, dt, seg, pred, n_pix=200, warm=16, batch=25):
    """D5: decode the predicted blocks after up to `warm` true latents of the
    episode (from latent 0 when it starts closer), pool to 64x64, RMSE in
    pixel/255 and PSNR, on the first n_pix segments.  With 16 latents of
    warm-up the decoded frames match the episode-aligned decode to an RMSE of
    0.0013 (tests/test_diffusion.py); with 4, 0.015."""
    T = seg["T"]
    groups = {}
    for i, (e, k) in enumerate(seg["pick"][:n_pix]):
        groups.setdefault(min(warm, k + 1), []).append(i)
    se, n = 0.0, 0
    for W, idx in groups.items():
        for j in range(0, len(idx), batch):
            ii = idx[j:j + batch]
            lat = np.stack([np.concatenate([dt.lat[seg["pick"][i][0], seg["pick"][i][1] - W + 1:seg["pick"][i][1] + 1],
                                            pred[i, 1:]]) for i in ii])
            fr = WD.decode_latents(vae, lat).cpu().numpy()
            fr = fr[:, 1 + 4 * (W - 1):]                                        # the predicted blocks' 4T frames
            p = pool(fr).astype(np.float64) / 255.0
            t = seg["frames"][ii].astype(np.float64) / 255.0
            se += ((p - t) ** 2).sum(); n += t.size
    mse = se / n
    return float(np.sqrt(mse)), float(10 * np.log10(1.0 / max(mse, 1e-12)))


# ---------------------------------------------------------------- M10 (D6) ------
def sdpa_math():
    """SDPA through its plain math kernel, which forward-mode autodiff supports."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    return sdpa_kernel(SDPBackend.MATH)


def m10_rof(den, dt, seg, S, n_states):
    """Reward observability (arXiv 2607.01736 eq. 7) with the 4-frame window as
    the state (D6).  f(window, a) = the window shifted by one block with the
    sampler's next latent appended (fixed noise, Euler without KV cache).
    Observable subspace span{C, C A, ..., C A^K}: vector-Jacobian products at
    the linearisation point.  Controllable subspace span{B, A B, ..., A^K B}:
    Jacobian-vector products (forward-mode autodiff, SDPA math kernel).  Both
    are exact derivatives of the same fp32 map.  K = H_plan blocks;
    the action at the linearisation point is zero, as for the other families."""
    dev = next(den.parameters()).device
    C_, ti, A_ = seg["C"], dt.ti, dt.td.act_dim
    Wp = torch.as_tensor(dt.probe.coef_, dtype=torch.float32, device=dev)          # (n_state, 3072)
    n_lat = 48 * 64
    rofs, kos, kcs = [], [], []
    for i in range(n_states):
        ctx0 = torch.as_tensor(seg["full"][i, :C_], device=dev)                    # (C, 48, 8, 8)
        cc = torch.as_tensor(seg["cond"][i, :C_], device=dev)[None]
        eps = torch.randn((1, 48, 1, 8, 8), generator=torch.Generator().manual_seed(1000 + i)).to(dev)
        a0 = torch.zeros(1, 1, BLOCK, A_, device=dev)

        def f(x_flat, a_blk):
            win = x_flat.view(1, C_, 48, 8, 8)
            z_new = _ode_next(den, win, cc, WD.cond_blocks(a_blk, ti).to(dev), eps, S.steps)
            return torch.cat([win[:, 1:], z_new[:, None]], 1).reshape(-1)

        x0 = ctx0.reshape(-1).clone()
        n = x0.numel()
        Cm = torch.zeros(Wp.shape[0], n, device=dev); Cm[:, (C_ - 1) * n_lat:] = Wp
        xr = ctx0[-1].clone().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(dt.rhead(xr[None])[0], xr)[0].reshape(-1)
        r = torch.zeros(n, device=dev); r[(C_ - 1) * n_lat:] = g
        with sdpa_math(), torch.enable_grad():
            _, vjp_fn = torch.func.vjp(lambda x: f(x, a0), x0)
        V = torch.linalg.qr(Cm.T)[0].T
        blk = Cm
        for _ in range(S.h_plan):
            with sdpa_math():           # create_graph=False: keep no backward graph per row
                blk = torch.stack([vjp_fn(row, retain_graph=True, create_graph=False)[0] for row in blk])
            blk = blk / blk.norm().clamp_min(1e-12)
            U, sv, Vt = torch.linalg.svd(torch.cat([V, blk]), full_matrices=False)
            V = Vt[sv > 1e-3 * sv[0]]
        rofs.append(float((V @ r).pow(2).sum() / r.pow(2).sum().clamp_min(1e-12)))
        kos.append(V.shape[0] / n)
        with sdpa_math(), torch.no_grad():   # forward-mode AD needs no autograd graph
            jvp_x = lambda v: torch.func.jvp(lambda x: f(x, a0), (x0,), (v,))[1]
            a0f = a0.reshape(-1)
            cols = [torch.func.jvp(lambda a: f(x0, a.view_as(a0)), (a0f,), (e_j,))[1]
                    for e_j in torch.eye(BLOCK * A_, device=dev)]
            blk = torch.stack(cols, 1)
            Wc = torch.linalg.qr(blk)[0]
            for _ in range(S.h_plan):
                blk = torch.stack([jvp_x(c) for c in blk.T], 1)
                blk = blk / blk.norm().clamp_min(1e-12)
                U, sv, Vt = torch.linalg.svd(torch.cat([Wc, blk], 1), full_matrices=False)
                Wc = U[:, sv > 1e-3 * sv[0]]
        kcs.append(Wc.shape[1] / n)
    return float(np.mean(rofs)), float(np.mean(kos)), float(np.mean(kcs))


def _ode_next(den, win, cc, fc, eps, steps):
    """Deterministic one-block map: Euler on the flow-matching ODE from fixed
    noise, the context clean in the same clip (chunks of 2 latent frames)."""
    C_ = win.shape[1]
    x_ctx = win.permute(0, 2, 1, 3, 4)
    z = eps.clone()
    ts = torch.linspace(1.0, 0.0, steps + 1, device=win.device)
    cond = torch.cat([cc, fc], 1)
    net = den.net
    # forward_with_cache takes any number of frames (forward() needs the
    # training clip length); its no_grad decorator is bypassed so M10 can
    # differentiate through it.
    fwc = getattr(type(net).forward_with_cache, "__wrapped__", type(net).forward_with_cache)
    for s in range(steps):
        x = torch.cat([x_ctx, z], 2)
        tf = torch.zeros(1, C_ + 1, device=win.device); tf[:, -1] = ts[s]
        v, _ = fwc(net, x, tf, cond, past_kv_list=None, current_position_offset=0, return_kv=False,
                   chunk_size=den.df_chunk_size)
        z = z - (ts[s + 1] - ts[s]) * v[:, :, -1:]
    return z[:, :, 0]


# ------------------------------------------------------------ interventions ------
def roots_blocks(S, seed=3):
    """D7: (episode 0..99, block k) with k >= ctx-1 and room for H_plan blocks."""
    rng = np.random.default_rng(seed)
    cand = [(e, k) for e in range(100) for k in range(max(S.ctx - 1, 5), N_LAT - 1 - S.h_plan)]
    return [cand[i] for i in rng.choice(len(cand), S.n_roots, replace=False)]


def interventions(samp, dt, S, rng):
    """M11 and M7 (evaluate.interventional_audits, in blocks)."""
    td = dt.td
    env = DMCEnv(dt.task, render=False); env.reset(0)
    H, A, C_ = S.h_plan, td.act_dim, S.ctx
    g_sup, g_mean, rhos, dsim, dwm = [], [], [], [], []
    for r_i, (e, k) in enumerate(roots_blocks(S)):
        root = td.mjstate_eval[e, 4 * k]
        ctx = dt.lat[e, k - C_ + 1:k + 1]; cc = dt.cond[e, k - C_ + 1:k + 1]
        logged = td.action[EV.N_TRAIN + e, 4 * k:4 * (k + H)]
        U = np.concatenate([np.clip(logged[None] + s * rng.standard_normal((16, 4 * H, A)), -1, 1)
                            for s in (0.1, 0.4)] + [rng.uniform(-1, 1, (32, 4 * H, A))]).astype(np.float32)
        J = np.array([-EV.replay(env, root, u)[1] for u in U])
        nz = torch.randn((len(U), H, 48, 8, 8), generator=torch.Generator().manual_seed(7000 + r_i))
        pred = samp(np.repeat(ctx[None], len(U), 0), torch.as_tensor(np.repeat(cc[None], len(U), 0)),
                    WD.cond_blocks(U.reshape(len(U), H, BLOCK, A), dt.ti), nz)
        Jh = -dt.block_reward(pred).sum(1).cpu().numpy()
        d = np.abs(Jh - J); g_sup.append(d.max()); g_mean.append(d.mean())
        rhos.append(spearmanr(Jh, J).statistic if np.std(J) > 0 and np.std(Jh) > 0 else np.nan)
        nb = S.m7_blocks
        for p_i in range(5):
            u1 = rng.uniform(-1, 1, (BLOCK * nb, A)).astype(np.float32); u2 = u1.copy()
            u2[0] = rng.uniform(-1, 1, A)
            o1, _ = EV.replay(env, root, u1); o2, _ = EV.replay(env, root, u2)
            nz2 = torch.randn((1, nb, 48, 8, 8), generator=torch.Generator().manual_seed(9000 + 10 * r_i + p_i)).repeat(2, 1, 1, 1, 1)
            zz = samp(np.repeat(ctx[None], 2, 0), torch.as_tensor(np.repeat(cc[None], 2, 0)),
                      WD.cond_blocks(np.stack([u1, u2]).reshape(2, nb, BLOCK, A), dt.ti), nz2)[:, -1].cpu().numpy()
            p = dt.probe.predict(zz)
            dsim.append(td.state(o1["qpos"], o1["qvel"]) - td.state(o2["qpos"], o2["qvel"]))
            dwm.append(p[0] - p[1])
    dsim, dwm = np.asarray(dsim), np.asarray(dwm)
    sd = dsim.std(0) + 1e-9; dsim, dwm = dsim / sd, dwm / sd
    cross = float(np.mean((dwm * dsim).sum(-1)))
    vw, vs = float(np.mean((dwm ** 2).sum(-1))), float(np.mean((dsim ** 2).sum(-1)))
    rho = np.asarray(rhos, float)
    return {"M11_g_sup": float(np.mean(g_sup)), "M11_mean_abs": float(np.mean(g_mean)),
            "M11_one_minus_rho": float(1 - np.nanmean(rho)) if np.isfinite(rho).any() else float("nan"),
            "M11_n_rho": int(np.isfinite(rho).sum()),
            "M7_gain": cross / vs, "M7_r2": cross ** 2 / (vw * vs), "M7_cosine": cross / np.sqrt(vw * vs)}


# ------------------------------------------------------------------ U1 (D9) ------
def u1_planning(samp, vae, dt, S, dev, seed=777):
    seeds = EV.U1_SEEDS[S.u1_first:S.u1_first + S.u1_episodes]
    E, A, H = len(seeds), dt.td.act_dim, S.h_plan
    cfg = PlanConfig(horizon=BLOCK * H, receding_horizon=BLOCK * S.replan_blocks, population=S.pop,
                     iterations=S.iters, action_low=-1.0, action_high=1.0)
    envs = [DMCEnv(dt.task, img=128) for _ in seeds]
    frames0 = np.stack([e.reset(s)["pixels"] for e, s in zip(envs, seeds)])
    enc = WD.StreamEncoder(vae, dev)
    lat_hist = [enc.reset(frames0).cpu().numpy()]                           # list of (E, 48, 8, 8)
    cond_hist = [WD.seed_cond(E, dt.ti).numpy()[:, 0]]
    gen = torch.Generator(device=dev).manual_seed(seed)
    ws = torch.zeros((E, BLOCK * H, A), dtype=torch.float64, device=dev)
    ret, decisions, tp, te = np.zeros(E), 0, 0.0, 0.0
    n_bad = n_all = 0
    while decisions < S.u1_T:
        t0 = time.time()
        C_ = min(S.ctx, len(lat_hist))
        ctx = np.stack(lat_hist[-C_:], 1); cc = np.stack(cond_hist[-C_:], 1)
        mu, std = ws.clone(), torch.full_like(ws, cfg.init_std)
        P = cfg.population
        for _ in range(cfg.iterations):
            seq = (mu[:, None] + std[:, None] * torch.randn((E, P) + mu.shape[1:], generator=gen, device=dev,
                                                           dtype=torch.float64)).clamp(-1, 1)
            fc = WD.cond_blocks(seq.reshape(E * P, H, BLOCK, A).float(), dt.ti)
            pred = samp(np.repeat(ctx, P, 0), torch.as_tensor(np.repeat(cc, P, 0)), fc, None)
            Jc = -dt.block_reward(pred).sum(1).double().reshape(E, P)
            fin = torch.isfinite(Jc)             # D11: a non-finite predicted cost gets weight zero
            n_bad += int((~fin).sum()); n_all += fin.numel()
            mn = torch.where(fin, Jc, torch.full_like(Jc, float("inf"))).min(1, keepdim=True).values
            mn = torch.where(torch.isfinite(mn), mn, torch.zeros_like(mn))
            w = torch.where(fin, torch.exp(-(torch.where(fin, Jc, mn) - mn) / max(cfg.temperature, 1e-8)),
                            torch.zeros_like(Jc))
            sw = w.sum(1, keepdim=True)          # no finite candidate at all: equal weights
            w = torch.where(sw > 0, w / sw.clamp_min(1e-300), torch.full_like(w, 1.0 / P))
            new_mu = torch.einsum("ep,epha->eha", w, seq)
            new_std = torch.sqrt(torch.einsum("ep,epha->eha", w, (seq - new_mu[:, None]) ** 2))
            mu = (1 - cfg.momentum) * new_mu + cfg.momentum * mu
            std = new_std.clamp_min(cfg.min_std)
        n_exec = BLOCK * S.replan_blocks
        act = mu[:, :n_exec].float().cpu().numpy()
        ws = torch.roll(mu, -n_exec, 1); ws[:, -n_exec:] = 0.0
        t1 = time.time(); tp += t1 - t0
        for b in range(S.replan_blocks):
            if decisions >= S.u1_T:          # an interval that does not divide the episode stops at its end
                break
            fr = []
            for j in range(BLOCK):
                out = [e.step(a) for e, a in zip(envs, act[:, BLOCK * b + j])]
                ret += np.array([r for _, r, _ in out]); fr.append(np.stack([o["pixels"] for o, _, _ in out]))
            lat_hist.append(enc.push(np.stack(fr, 1)).cpu().numpy())
            cond_hist.append(WD.cond_blocks(act[:, None, BLOCK * b:BLOCK * (b + 1)], dt.ti).numpy()[:, 0])
            decisions += BLOCK
        te += time.time() - t1
    return ret, {"U1_decisions": decisions, "t_u1_plan_s": tp, "t_u1_env_s": te,
                 "U1_nonfinite_frac": n_bad / max(n_all, 1)}


# ------------------------------------------------------------------ U2 (D10) ------
def u2_imagined(samp, vae, dt, S, dev):
    seeds = U2P.U2_SEEDS[:S.u2_seeds]
    prior = U2P.load_prior(dt.task, dev)
    dr = U2P.crn_draws(prior.act_dim, seeds, S.u2_H)
    n_pol = len(U2P.POLICIES)
    env = DMCEnv(dt.task, img=128)
    f0 = np.stack([env.reset(s)["pixels"] for s in seeds])
    z0 = WD.encode_video(vae, f0[:, None]).cpu().numpy()[:, 0]            # (S, 48, 8, 8)
    Z = np.tile(z0, (n_pol, 1, 1, 1))                                       # policy-major rows
    hist, chist = [Z], [np.repeat(WD.seed_cond(1, dt.ti).numpy()[:, 0], len(Z), 0)]
    R = np.zeros(len(Z))
    for b in range(S.u2_H // BLOCK):
        base = prior.act(dt.probe.proprio(hist[-1]).astype(np.float32))
        a = np.stack([U2P.policy_actions(base, dr, BLOCK * b + j) for j in range(BLOCK)], 1)   # (N, 4, A)
        C_ = min(S.ctx, len(hist))
        fc = WD.cond_blocks(a[:, None], dt.ti)
        nz = torch.randn((len(Z), 1, 48, 8, 8), generator=torch.Generator().manual_seed(5000 + b))
        z = samp(np.stack(hist[-C_:], 1), torch.as_tensor(np.stack(chist[-C_:], 1)), fc, nz)[:, 0]
        R += dt.block_reward(z).cpu().numpy()
        hist.append(z.cpu().numpy()); chist.append(fc.numpy()[:, 0])
    return R.reshape(n_pol, -1).mean(1)


# --------------------------------------------------------------------- main ------
_VAE = {}


def evaluate(ckpt_path, task, dev, S=Settings(), parts=("metrics", "u1", "u2")):
    t0 = time.time()
    den, ck = WD.load_denoiser(ckpt_path, dev)
    den.requires_grad_(False)          # M10 differentiates with respect to inputs only
    if "v" not in _VAE:
        _VAE["v"] = WD.load_vae(dev)
    vae = _VAE["v"]
    dt = DiffTask(task, dev)
    samp = WD.Sampler(den, steps=S.steps, ardiff=S.ardiff, cfg_scale=S.cfg_scale, micro_batch=S.micro_batch)
    out = {k: ck[k] for k in ("family", "run", "seed", "data_frac", "step", "frac", "val_loss")}
    out.update(task=task, width=float("nan"), arch="miniworld_0.5B", size=None, ckpt=str(ckpt_path),
               gpu=torch.cuda.get_device_name(0), eval_settings=dataclasses.asdict(S), block=BLOCK)
    out.update({f"probe_r2_heldout_{k}": v for k, v in dt.heads["probe_r2_heldout"].items()})
    out["reward_head_r2_heldout"] = dt.heads["reward_r2_heldout"]
    if "metrics" in parts:
        seg = block_segments(dt, S)
        N, T = len(seg["full"]), seg["T"]
        noise = torch.randn((N, T, 48, 8, 8), generator=torch.Generator().manual_seed(11)).numpy()
        z_true = seg["full"][:, seg["C"] - 1:]                                   # (N, T+1, ...)
        D = int(np.prod(z_true.shape[2:]))
        flat = lambda x: x.reshape(*x.shape[:2], -1).astype(np.float64)
        var = float(flat(z_true[:, 1:]).reshape(-1, D).var(0).sum())
        st_true = dt.probe.predict(z_true[:, 1:].reshape(-1, *z_true.shape[2:])).reshape(N, T, -1)
        preds = {}
        for pname, R in (("P1", 1), ("P3", S.h_plan), ("P2", INF)):
            pr = predict_resync(samp, seg, R, noise); preds[pname] = pr
            raw = ((flat(pr[:, 1:]) - flat(z_true[:, 1:])) ** 2).sum(-1)
            out[f"latent_err_{pname}_raw"] = float(raw.mean())
            out[f"latent_err_{pname}_normalised"] = float(raw.mean() / max(var, 1e-12))
            sp = dt.probe.predict(pr[:, 1:].reshape(-1, *pr.shape[2:])).reshape(N, T, -1)
            out[f"latent_err_{pname}_probe"] = float(((sp - st_true) ** 2).sum(-1).mean())
            out[f"obs_rmse_{pname}"], out[f"psnr_{pname}"] = pixel_errors(vae, dt, seg, pr, n_pix=S.n_pix)
            out[f"probe_rmse_{pname}"] = float(np.sqrt(((sp - seg["states"][:, 1:]) ** 2).mean()))
        p2 = preds["P2"]
        for kb in sorted({1, S.h_plan // 2, S.h_plan}):           # blocks; 1, 4, 8 = 4, 16, 32 decisions
            p = dt.probe.predict(p2[:, kb]); tg = seg["states"][:, kb]
            out[f"M8_probe_r2_k{BLOCK * kb}"] = float(1 - ((p - tg) ** 2).sum(-1).mean() / max(tg.var(0).sum(), 1e-12))
        sub = {k: (v[:S.m9_n] if isinstance(v, np.ndarray) else v) for k, v in seg.items()}
        draws = []
        for d in range(8):
            nz = torch.randn((min(S.m9_n, N), S.h_plan, 48, 8, 8), generator=torch.Generator().manual_seed(300 + d))
            draws.append(samp(sub["full"][:, :sub["C"]], torch.as_tensor(sub["cond"][:, :sub["C"]]),
                              torch.as_tensor(sub["cond"][:, sub["C"]:sub["C"] + S.h_plan]), nz)[:, -1].cpu().numpy())
        out[f"M9_sample_disagreement_k{BLOCK * S.h_plan}"] = float(np.stack(draws).reshape(8, len(draws[0]), -1).var(0).sum(-1).mean())
        out["M10_ROF"], out["M10_k_obs"], out["M10_k_ctrl"] = m10_rof(den, dt, seg, S, min(S.m10_n, N))
        out["M10_CROF"] = out["M10_ROF"] + (1 - out["M10_k_ctrl"]) + (1 - out["M10_k_obs"]) + out["probe_rmse_P2"]
        imag = dt.probe.predict(p2[:, :S.h_plan + 1].reshape(-1, *p2.shape[2:])).reshape(N, S.h_plan + 1, -1)
        real = seg["states"][:, :S.h_plan + 1]
        out["M12_ikce_model"], out["M12_ikce_real"] = EV.ikce(imag, dt.td4), EV.ikce(real, dt.td4)
        out["M12_ikce_log_ratio"] = float(abs(np.log(out["M12_ikce_model"] / out["M12_ikce_real"])))
        out["M12_ikce_root_model"] = EV.ikce_root(imag, dt.td4, task)
        out["M12_ikce_root_real"] = EV.ikce_root(real, dt.td4, task)
        out["M12_ikce_root_log_ratio"] = float(abs(np.log(max(out["M12_ikce_root_model"], 1e-12)
                                                          / max(out["M12_ikce_root_real"], 1e-12))))
        out.update(interventions(samp, dt, S, np.random.default_rng(12345)))
    out["t_metrics_s"] = time.time() - t0
    if "u1" in parts:
        t1 = time.time()
        if S.u1_first:                 # D12: a chunk of extra episodes draws its own planner and sampler noise
            torch.manual_seed(1000 + S.u1_first)
        ret, info = u1_planning(samp, vae, dt, S, dev, seed=777 + S.u1_first)
        out["U1_seeds"] = EV.U1_SEEDS[S.u1_first:S.u1_first + S.u1_episodes]
        out["U1_return"] = float(ret.mean()); out["U1_returns"] = ret.tolist(); out.update(info)
        out["t_u1_s"] = time.time() - t1
    if "u2" in parts:
        t2 = time.time()
        u2_default = (S.u2_seeds, S.u2_H) == (Settings.u2_seeds, Settings.u2_H)
        true_r = U2P.true_returns(task, U2P.U2_SEEDS[:S.u2_seeds], S.u2_H, cache=u2_default)[0]
        imag_r = u2_imagined(samp, vae, dt, S, dev)
        out["U2_kendall_tau"] = float(kendalltau(imag_r, true_r).statistic)
        out["U2_imagined"] = imag_r.tolist(); out["U2_true"] = true_r.tolist()
        out["t_u2_s"] = time.time() - t2
    out["t_total_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--tasks", default=",".join(WD.TASKS))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--parts", default="metrics,u1,u2")
    ap.add_argument("--out", help="result directory; required with --tiny/--set/--parts")
    a = ap.parse_args()
    dev = "cuda"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    S = Settings(**TINY) if a.tiny else Settings()
    for kv in a.set:
        k, v = kv.split("=")
        setattr(S, k, type(getattr(S, k))(v))
    parts = tuple(a.parts.split(","))
    official = S == Settings() and set(parts) == {"metrics", "u1", "u2"} and not a.out
    if not official and not a.out:
        ap.error("non-default settings or parts need --out")
    failed = []
    for c in map(pathlib.Path, a.ckpts):
        run = c.parent.name
        for task in a.tasks.split(","):
            res_dir = (RESULTS / "dmc" / "diffusion" / task / run) if official else pathlib.Path(a.out) / task / run
            f = res_dir / (c.stem + ".json")
            if f.exists() and not os.environ.get("FORCE"):
                print("skip (done)", f, flush=True); continue
            try:
                r = evaluate(c, task, dev, S, parts)
            except Exception as e:
                traceback.print_exc(); failed.append(f"{c}:{task}"); continue
            res_dir.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f".tmp{os.getpid()}"); json.dump(r, open(tmp, "w"), indent=1); tmp.replace(f)
            print(f"diffusion/{task}/{run}/{c.stem}: U1={r.get('U1_return', float('nan')):.1f} "
                  f"U2_tau={r.get('U2_kendall_tau', float('nan')):.3f} val={r['val_loss']:.4f} "
                  f"latP1={r.get('latent_err_P1_raw', float('nan')):.2f} "
                  f"({r['t_metrics_s']:.0f}s metrics, {r.get('t_u1_s', 0):.0f}s U1, {r.get('t_u2_s', 0):.0f}s U2)",
                  flush=True)
    if failed:
        sys.exit(f"{len(failed)} evaluation(s) failed: {failed}")


if __name__ == "__main__":
    main()
