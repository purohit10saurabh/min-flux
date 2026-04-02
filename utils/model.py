"""
Shared model building blocks for FLUX.1 and FLUX.2.

Contains: sinusoidal timestep embeddings, RoPE positional embeddings,
multi-axis position embedding module, adaptive layer norm, and joint attention.

References (source of truth):
1) diffusers embeddings — get_timestep_embedding, TimestepEmbedding, get_1d_rotary_pos_embed, apply_rotary_emb:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
2) diffusers normalization — AdaLayerNormContinuous:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py
3) diffusers transformer_flux — FluxPosEmbed, FluxAttnProcessor:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
4) diffusers transformer_flux2 — Flux2PosEmbed, Flux2AttnProcessor:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_channels, out_channels, bias=bias)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0, freqs_dtype=torch.float32):
    if isinstance(pos, int):
        pos = torch.arange(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    freqs = torch.outer(pos.float(), freqs)
    return freqs.cos().repeat_interleave(2, dim=1).float(), freqs.sin().repeat_interleave(2, dim=1).float()


def apply_rotary_emb(x, freqs_cis):
    cos, sin = freqs_cis
    cos = cos[None, :, None, :].to(x.device)
    sin = sin[None, :, None, :].to(x.device)
    x_real, x_imag = rearrange(x, '... (d two) -> ... d two', two=2).unbind(-1)
    x_rotated = rearrange(torch.stack([-x_imag, x_real], dim=-1), 'b s h d two -> b s h (d two)')
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class PosEmbed(nn.Module):
    def __init__(self, theta, axes_dim):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        cos_out, sin_out = [], []
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(self.axes_dim[i], ids[..., i].float(), theta=self.theta)
            cos_out.append(cos)
            sin_out.append(sin)
        return torch.cat(cos_out, dim=-1).to(ids.device), torch.cat(sin_out, dim=-1).to(ids.device)


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim, conditioning_dim, elementwise_affine=False, eps=1e-6, bias=True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, dim * 2, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x, conditioning):
        emb = self.linear(self.silu(conditioning).to(x.dtype))
        scale, shift = emb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


def joint_attention(q, k, v, q_ctx, k_ctx, v_ctx, image_rotary_emb=None):
    q = torch.cat([q_ctx, q], dim=1) if q_ctx is not None else q
    k = torch.cat([k_ctx, k], dim=1) if k_ctx is not None else k
    v = torch.cat([v_ctx, v], dim=1) if v_ctx is not None else v
    if image_rotary_emb is not None:
        q = apply_rotary_emb(q, image_rotary_emb)
        k = apply_rotary_emb(k, image_rotary_emb)
    q, k, v = (rearrange(t, 'b s h d -> b h s d') for t in (q, k, v))
    return rearrange(F.scaled_dot_product_attention(q, k, v), 'b h s d -> b s h d')
