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

import torch
import torch.nn as nn
from einops import rearrange
from utils.model import (
    get_timestep_embedding, TimestepEmbedding,
    PosEmbed, AdaLayerNormContinuous, joint_attention,
)


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


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
        h = self.heads
        q = self.norm_q(rearrange(self.to_q(x), 'b s (h d) -> b s h d', h=h))
        k = self.norm_k(rearrange(self.to_k(x), 'b s (h d) -> b s h d', h=h))
        v = rearrange(self.to_v(x), 'b s (h d) -> b s h d', h=h)
        q_ctx = self.norm_added_q(rearrange(self.add_q_proj(ctx), 'b s (h d) -> b s h d', h=h))
        k_ctx = self.norm_added_k(rearrange(self.add_k_proj(ctx), 'b s (h d) -> b s h d', h=h))
        v_ctx = rearrange(self.add_v_proj(ctx), 'b s (h d) -> b s h d', h=h)
        out = rearrange(joint_attention(q, k, v, q_ctx, k_ctx, v_ctx, image_rotary_emb), 'b s h d -> b s (h d)')
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
        h = self.heads
        q = self.norm_q(rearrange(self.to_q(x), 'b s (h d) -> b s h d', h=h))
        k = self.norm_k(rearrange(self.to_k(x), 'b s (h d) -> b s h d', h=h))
        v = rearrange(self.to_v(x), 'b s (h d) -> b s h d', h=h)
        return rearrange(joint_attention(q, k, v, None, None, None, image_rotary_emb), 'b s h d -> b s (h d)')


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

        self.pos_embed = PosEmbed(theta=10000, axes_dim=axes_dims_rope)
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
