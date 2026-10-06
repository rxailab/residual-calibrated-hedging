"""Minimal stand-in for the flash-attn package, for MiniWorld (third_party/MiniWorld).

MiniWorld calls flash_attn_func only on its unmasked, cache-free attention path;
every masked or KV-cached call (diffusion-forcing training, streaming
inference) already goes through torch's scaled_dot_product_attention.  This
shim routes the remaining path through SDPA as well, with flash-attn's
(B, N, heads, head_dim) layout, so the code runs without building flash-attn
and on GPUs it does not support (V100).  The arithmetic is the same
attention; only the kernel differs.
"""
import torch
import torch.nn.functional as F

__version__ = "shim-sdpa"


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                         dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2)
