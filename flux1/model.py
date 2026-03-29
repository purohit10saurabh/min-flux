"""
Minimal Flux (FLUX.1) transformer model — the complete architecture.

References (source of truth):
1) diffusers FluxTransformer2DModel — FluxTransformerBlock, FluxSingleTransformerBlock, FluxPosEmbed:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
2) diffusers embeddings — Timesteps, TimestepEmbedding, CombinedTimestepGuidanceTextProjEmbeddings:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
3) diffusers normalization — AdaLayerNormZero, AdaLayerNormZeroSingle, AdaLayerNormContinuous:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py
4) Original Flux implementation (Black Forest Labs):
   https://github.com/black-forest-labs/flux
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rotary import apply_rotary_emb, get_1d_rotary_pos_embed


# --- Sinusoidal Timestep Embeddings (embeddings.py:26-77, 1261-1306, 1309-1325) ---

def get_timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_channels, out_channels)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


# --- Combined Timestep + Guidance + Text Embeddings (embeddings.py:1584-1624) ---

class FluxTimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim, guidance_embeds=False):
        super().__init__()
        self.time_proj = lambda t: get_timestep_embedding(t, 256)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.guidance_embedder = TimestepEmbedding(256, embedding_dim) if guidance_embeds else None
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim)

    def forward(self, timestep, guidance, pooled_projection):
        t_emb = self.timestep_embedder(self.time_proj(timestep).to(pooled_projection.dtype))
        if self.guidance_embedder is not None and guidance is not None:
            g_emb = self.guidance_embedder(self.time_proj(guidance).to(pooled_projection.dtype))
            t_emb = t_emb + g_emb
        return t_emb + self.text_embedder(pooled_projection)


# --- Adaptive Layer Norms (normalization.py:130-202, 307-351) ---

class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 3 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim, conditioning_dim, elementwise_affine=False, eps=1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x, conditioning):
        emb = self.linear(self.silu(conditioning).to(x.dtype))
        scale, shift = emb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


# --- RoPE Positional Embedding (transformer_flux.py:494-522) ---

class FluxPosEmbed(nn.Module):
    def __init__(self, theta=10000, axes_dim=(16, 56, 56)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        cos_out, sin_out = [], []
        for i in range(ids.shape[-1]):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i], ids[:, i].float(), theta=self.theta,
                repeat_interleave_real=True, use_real=True,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        return torch.cat(cos_out, dim=-1).to(ids.device), torch.cat(sin_out, dim=-1).to(ids.device)


# --- Joint Attention (transformer_flux.py:75-140, 275-352 simplified) ---

def flux_attention(q, k, v, q_ctx, k_ctx, v_ctx, image_rotary_emb=None):
    q = torch.cat([q_ctx, q], dim=1) if q_ctx is not None else q
    k = torch.cat([k_ctx, k], dim=1) if k_ctx is not None else k
    v = torch.cat([v_ctx, v], dim=1) if v_ctx is not None else v
    if image_rotary_emb is not None:
        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


class FluxJointAttention(nn.Module):
    def __init__(self, dim, heads=24, head_dim=128):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=True), nn.Dropout(0.0))
        self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.add_q_proj = nn.Linear(dim, inner_dim)
        self.add_k_proj = nn.Linear(dim, inner_dim)
        self.add_v_proj = nn.Linear(dim, inner_dim)
        self.to_add_out = nn.Linear(inner_dim, dim)

    def forward(self, x, ctx, image_rotary_emb=None):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k = self.norm_k(self.to_k(x).unflatten(-1, (self.heads, -1)))
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        q_ctx = self.norm_added_q(self.add_q_proj(ctx).unflatten(-1, (self.heads, -1)))
        k_ctx = self.norm_added_k(self.add_k_proj(ctx).unflatten(-1, (self.heads, -1)))
        v_ctx = self.add_v_proj(ctx).unflatten(-1, (self.heads, -1))
        out = flux_attention(q, k, v, q_ctx, k_ctx, v_ctx, image_rotary_emb)
        out = out.flatten(-2)
        txt_len = ctx.shape[1]
        return self.to_add_out(out[:, :txt_len]), self.to_out(out[:, txt_len:])


class FluxSingleAttention(nn.Module):
    def __init__(self, dim, heads=24, head_dim=128):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)

    def forward(self, x, image_rotary_emb=None):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k = self.norm_k(self.to_k(x).unflatten(-1, (self.heads, -1)))
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        out = flux_attention(q, k, v, None, None, None, image_rotary_emb)
        return out.flatten(-2)


# --- Transformer Blocks (transformer_flux.py:355-491) ---

class FluxTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, eps=1e-6):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn = FluxJointAttention(dim, heads=num_heads, head_dim=head_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim))
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim))

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None):
        norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        norm_ctx, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, temb)

        ctx_attn, attn = self.attn(norm_x, norm_ctx, image_rotary_emb)

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn
        norm_ff = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_ff)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * ctx_attn
        norm_ctx_ff = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_ctx_ff)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
        self.attn = FluxSingleAttention(dim, heads=num_heads, head_dim=head_dim)

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None):
        txt_len = encoder_hidden_states.shape[1]
        x = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = x
        norm_x, gate = self.norm(x, temb)
        mlp_out = self.act_mlp(self.proj_mlp(norm_x))
        attn_out = self.attn(norm_x, image_rotary_emb)
        x = residual + gate.unsqueeze(1) * self.proj_out(torch.cat([attn_out, mlp_out], dim=2))
        return x[:, :txt_len], x[:, txt_len:]


# --- Main Model (transformer_flux.py:525-778) ---

class FluxTransformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels=64,
        num_layers=19,
        num_single_layers=38,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=False,
        axes_dims_rope=(16, 56, 56),
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.inner_dim = inner_dim
        self.out_channels = in_channels
        self.guidance_embeds = guidance_embeds

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        self.time_text_embed = FluxTimestepEmbedding(inner_dim, pooled_projection_dim, guidance_embeds)
        self.context_embedder = nn.Linear(joint_attention_dim, inner_dim)
        self.x_embedder = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(inner_dim, num_attention_heads, attention_head_dim) for _ in range(num_layers)
        ])
        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(inner_dim, num_attention_heads, attention_head_dim) for _ in range(num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=True)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance=None,
    ):
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        image_rotary_emb = self.pos_embed(torch.cat((txt_ids, img_ids), dim=0))

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb, image_rotary_emb)

        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb, image_rotary_emb)

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


if __name__ == "__main__":
    print("Flux Transformer Model")
    print("=" * 40)
    model = FluxTransformer2DModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e9:.2f}B")
    print(f"\nArchitecture: {model.inner_dim}d, "
          f"{len(model.transformer_blocks)} double + {len(model.single_transformer_blocks)} single blocks")
    print("\nComponents (bottom-up):")
    print("1. FluxPosEmbed - RoPE positional encoding")
    print("2. FluxTimestepEmbedding - timestep + guidance + pooled text")
    print("3. AdaLayerNormZero/Single/Continuous - adaptive modulation")
    print("4. FluxJointAttention - double-stream joint attention (text + image)")
    print("5. FluxSingleAttention - single-stream self-attention")
    print("6. FluxTransformerBlock - double-stream block")
    print("7. FluxSingleTransformerBlock - single-stream block")
    print("8. FluxTransformer2DModel - full model")
