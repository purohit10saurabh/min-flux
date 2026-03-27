"""
Rotary positional embeddings (RoPE) for Flux transformers.

Source of truth: diffusers embeddings module
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
  get_1d_rotary_pos_embed (lines 1119-1183) — Flux path: use_real=True, repeat_interleave_real=True
  apply_rotary_emb (lines 1186-1240) — Flux path: use_real=True, unbind_dim=-1, sequence_dim=1
"""

import torch


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0, repeat_interleave_real=True, use_real=True, freqs_dtype=torch.float32):
    if isinstance(pos, int):
        pos = torch.arange(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    freqs = torch.outer(pos.float(), freqs)
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
    return freqs_cos, freqs_sin


def apply_rotary_emb(x, freqs_cis, sequence_dim=1):
    cos, sin = freqs_cis
    if sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    cos, sin = cos.to(x.device), sin.to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
