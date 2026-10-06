"""The video-diffusion family on DMControl: a fine-tuned MiniWorld-0.5B
checkpoint (experiments/dmc/train_diffusion.py) and the frozen Wan2.2 VAE.

Time base.  A block is 4 decisions.  Each episode is one causal video: latent
frame 0 encodes frame 0 alone and latent frame k >= 1 encodes frames
4k-3 .. 4k, which decisions 4k-4 .. 4k-1 produce.  The model predicts latent
frame k+1 from the latent history and that block's 4 actions.  "The state at
latent k" is the state at frame 4k.

The VAE always runs in fp32 (as when D was pre-encoded), outside the DiT's
autocast, so the latents a checkpoint is given at evaluation are the latents
it was trained on.  StreamEncoder encodes a growing video one block at a time
with the encoder's causal feature cache, which is exactly the whole-video
encoding (tests/test_diffusion.py).
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "third_party" / "MiniWorld", ROOT / "third_party" / "shims"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np
import torch
import miniworld.denoiser as _MD
from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
from miniworld.vae import load_wan22_vae
from miniworld.vae.wan22_vae import patchify

_MD._print0 = lambda *a, **k: None          # the sampler logs every chunk commit
TASKS = ("cartpole-swingup", "walker-walk", "cup-catch")
A_SLOT, BLOCK = 7, 4
COND_DIM = BLOCK * A_SLOT + len(TASKS)
VAE_PATH = "/scratch/hpc/11/xiar3/MetricTrust/pretrained/Wan2.2_VAE.pth"
LAT_C, LAT_H = 48, 8


def amp_dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability(0) >= (8, 0) else torch.float16


def load_denoiser(ckpt_path, device="cuda"):
    """EMA weights of a train_diffusion.py snapshot, as an fp32 module (the
    sampler runs it under autocast)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = DenoiserConfig(**{k: v for k, v in ck["config"].items()})
    den = build_denoiser_from_mode(cfg)
    den.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    return den.to(device).eval(), ck


def load_vae(device="cuda"):
    return load_wan22_vae({"vae_checkpoint": VAE_PATH})


# --- conditioning ---------------------------------------------------------------
def cond_blocks(actions, task_idx):
    """(N, F, 4, A) actions of F blocks -> (N, F, COND_DIM): each decision's
    action zero-padded into DROID's 7 slots, then the task one-hot
    (experiments/dmc/train_diffusion.py design note 1)."""
    a = torch.as_tensor(actions, dtype=torch.float32)
    N, F, b, A = a.shape
    pad = torch.zeros(N, F, BLOCK, A_SLOT, dtype=torch.float32, device=a.device)
    pad[..., :A] = a
    out = torch.zeros(N, F, COND_DIM, dtype=torch.float32, device=a.device)
    out[..., :BLOCK * A_SLOT] = pad.reshape(N, F, BLOCK * A_SLOT)
    out[..., BLOCK * A_SLOT + task_idx] = 1.0
    return out


def seed_cond(N, task_idx, device="cpu"):
    """Condition of an episode's latent frame 0 (no preceding action)."""
    c = torch.zeros(N, 1, COND_DIM, device=device)
    c[..., BLOCK * A_SLOT + task_idx] = 1.0
    return c


# --- VAE -----------------------------------------------------------------------------
def _video(frames, device):
    """(B, T, 128, 128, 3) uint8 -> (B, 3, T, 128, 128) in [-1, 1], fp32."""
    v = torch.as_tensor(np.ascontiguousarray(frames), device=device).float() / 127.5 - 1.0
    return v.permute(0, 4, 1, 2, 3)


@torch.no_grad()
def encode_video(vae, frames, device="cuda"):
    """(B, T, 128, 128, 3) uint8, T = 1 + 4n, from an episode start ->
    (B, 1 + n, 48, 8, 8) fp32 latents."""
    with torch.autocast("cuda", enabled=False):
        lat = vae.model.encode(_video(frames, device), vae.scale).float()
    return lat.permute(0, 2, 1, 3, 4)


@torch.no_grad()
def decode_latents(vae, lat):
    """(B, K, 48, 8, 8) latents of an episode prefix (latent 0 first) ->
    (B, 1 + 4(K-1), 128, 128, 3) uint8 frames."""
    z = torch.as_tensor(lat, device="cuda").float().permute(0, 2, 1, 3, 4)
    with torch.autocast("cuda", enabled=False):
        v = vae.model.decode(z, vae.scale).float().clamp(-1, 1)
    return ((v.permute(0, 2, 3, 4, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)


class StreamEncoder:
    """Encode an episode block by block, keeping the encoder's causal feature
    cache between calls: reset(frame 0) returns latent 0, each push(4 frames)
    the next latent.  Identical to encode_video on the whole prefix."""

    def __init__(self, vae, device="cuda"):
        self.m, self.scale, self.dev = vae.model, vae.scale, device
        self.cache = None

    @torch.no_grad()
    def _run(self, frames):
        m = self.m
        m._enc_feat_map, m._enc_conv_idx = self.cache, [0]
        with torch.autocast("cuda", enabled=False):
            out = m.encoder(patchify(_video(frames, self.dev), patch_size=2),
                            feat_cache=m._enc_feat_map, feat_idx=m._enc_conv_idx)
            mu, _ = m.conv1(out).chunk(2, dim=1)
        s0, s1 = self.scale
        mu = (mu - s0.view(1, m.z_dim, 1, 1, 1)) * s1.view(1, m.z_dim, 1, 1, 1)
        self.cache = m._enc_feat_map
        return mu.float()[:, :, -1]                       # (B, 48, 8, 8)

    def reset(self, frame0):
        """frame0: (B, 128, 128, 3) uint8."""
        self.m.clear_cache()
        self.cache = self.m._enc_feat_map
        return self._run(np.asarray(frame0)[:, None])

    def push(self, frames4):
        """frames4: (B, 4, 128, 128, 3) uint8, the frames of the next block."""
        return self._run(frames4)


# --- generation --------------------------------------------------------------------------
class Sampler:
    """Block-wise generation with MiniWorld's streaming AR-diffusion sampler
    (DiffusionForcingDenoiser.generate_eval_latents_streaming): the context
    frames prefill the KV cache, the future frames are denoised in chunks of
    2 latent frames with the AR-diffusion offset between chunks.  The active
    window is (cache + in-flight) chunks x 2 <= the trained 16 frames."""

    def __init__(self, den, steps=8, ardiff=2, cfg_scale=1.0, max_cache_chunks=4, inflight_chunks=4,
                 micro_batch=128):
        self.den, self.steps, self.ardiff, self.cfg_scale = den, steps, ardiff, cfg_scale
        self.max_cache, self.inflight, self.mb = max_cache_chunks, inflight_chunks, micro_batch
        assert 2 * (max_cache_chunks + inflight_chunks) <= den.trained_num_frames

    @torch.no_grad()
    def __call__(self, ctx, ctx_cond, fut_cond, noise=None):
        """ctx (B, W, 48, 8, 8), ctx_cond (B, W, C), fut_cond (B, F, C), noise
        (B, F, 48, 8, 8) or None -> (B, F, 48, 8, 8) fp32 sampled latents."""
        den = self.den
        den.steps, den.df_ardiff_step, den.cfg_scale = self.steps, self.ardiff, self.cfg_scale
        dev = next(den.parameters()).device
        ctx = torch.as_tensor(ctx, dtype=torch.float32, device=dev)
        B, W = ctx.shape[:2]
        F_ = fut_cond.shape[1]
        assert W <= 2 * self.max_cache, "context longer than the KV cache"
        out = []
        for i in range(0, B, self.mb):
            sl = slice(i, i + self.mb)
            b = ctx[sl].shape[0]
            lat = torch.cat([ctx[sl], torch.zeros(b, F_, LAT_C, LAT_H, LAT_H, device=dev)], 1).permute(0, 2, 1, 3, 4)
            cond = torch.cat([torch.as_tensor(ctx_cond[sl], device=dev),
                              torch.as_tensor(fut_cond[sl], device=dev)], 1).float()
            nz = None
            if noise is not None:
                nz = torch.cat([torch.zeros(b, W, LAT_C, LAT_H, LAT_H, device=dev),
                                torch.as_tensor(noise[sl], device=dev).float()], 1).permute(0, 2, 1, 3, 4)
            out.append(self._run(lat, cond, W, F_, nz))
        return torch.cat(out)

    def _run(self, lat, cond, W, F_, nz):
        """One micro-batch; if it runs out of GPU memory, the same sequences in halves.  With
        fixed noise the result does not depend on the split (only the arithmetic's batch shape)."""
        try:
            with torch.autocast("cuda", dtype=amp_dtype()):
                z = self.den.generate_eval_latents_streaming(
                    lat, cond, total_len=W + F_, history_len=W, max_cache_chunks=self.max_cache,
                    inflight_chunks=self.inflight, noise=nz)
            return z[:, :, W:].float().permute(0, 2, 1, 3, 4)
        except torch.OutOfMemoryError:
            b = lat.shape[0]
            if b == 1:
                raise
            torch.cuda.empty_cache()
            h = b // 2
            print(f"[Sampler] out of memory at {b} sequences x {W + F_} frames: splitting in two", flush=True)
            return torch.cat([self._run(lat[:h], cond[:h], W, F_, None if nz is None else nz[:h]),
                              self._run(lat[h:], cond[h:], W, F_, None if nz is None else nz[h:])])
