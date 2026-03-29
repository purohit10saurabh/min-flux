"""
Minimal Flux2 (FLUX.2) transformer model — the complete architecture.
Key differences from FLUX.1 (flux1/model.py):
- Shared modulation (one set per stream type, not per-block AdaLayerNorm)
- SwiGLU activation (not GELU-approximate)
- Fused QKV+MLP single-stream blocks (parallel transformer)
- No pooled_projections in timestep embedding
- RoPE: theta=2000, axes_dim=(32,32,32,32), separate text/image embeddings
- All biases=False

References (source of truth):
1) diffusers Flux2Transformer2DModel — blocks, modulation, attention:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py
2) diffusers embeddings — Timesteps, TimestepEmbedding:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
3) diffusers normalization — AdaLayerNormContinuous:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rotary import apply_rotary_emb, get_1d_rotary_pos_embed


# --- Sinusoidal Timestep Embeddings (embeddings.py:26-77, 1261-1306) ---

def get_timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_channels, out_channels, bias=bias)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


# --- Flux2 Timestep + Guidance Embedding (transformer_flux2.py:982-1014) ---

class Flux2TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, guidance_embeds=True):
        super().__init__()
        self.time_proj = lambda t: get_timestep_embedding(t, 256)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.guidance_embedder = TimestepEmbedding(256, embedding_dim) if guidance_embeds else None

    def forward(self, timestep, guidance):
        t_emb = self.timestep_embedder(self.time_proj(timestep).to(timestep.dtype))
        if self.guidance_embedder is not None and guidance is not None:
            g_emb = self.guidance_embedder(self.time_proj(guidance).to(guidance.dtype))
            return t_emb + g_emb
        return t_emb


# --- Shared Modulation (transformer_flux2.py:1017-1037) ---

class Flux2Modulation(nn.Module):
    def __init__(self, dim, mod_param_sets=2):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.act_fn = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 3 * mod_param_sets, bias=False)

    def forward(self, temb):
        return self.linear(self.act_fn(temb))

    @staticmethod
    def split(mod, mod_param_sets):
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
        return tuple(params[3 * i: 3 * (i + 1)] for i in range(mod_param_sets))


# --- Output Norm (normalization.py:307-351) ---

class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim, conditioning_dim, elementwise_affine=False, eps=1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x, conditioning):
        emb = self.linear(self.silu(conditioning).to(x.dtype))
        scale, shift = emb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


# --- RoPE (transformer_flux2.py:950-979) ---

class Flux2PosEmbed(nn.Module):
    def __init__(self, theta=2000, axes_dim=(32, 32, 32, 32)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        cos_out, sin_out = [], []
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i], ids[..., i].float(), theta=self.theta,
                repeat_interleave_real=True, use_real=True,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        return torch.cat(cos_out, dim=-1).to(ids.device), torch.cat(sin_out, dim=-1).to(ids.device)


# --- SwiGLU FFN (transformer_flux2.py:283-322) ---

class Flux2SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Flux2FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=3.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=False)

    def forward(self, x):
        return self.linear_out(self.act_fn(self.linear_in(x)))


# --- Joint Attention for double-stream (transformer_flux2.py:325-391, 493-548) ---

def flux2_attention(q, k, v, q_ctx, k_ctx, v_ctx, image_rotary_emb=None):
    q = torch.cat([q_ctx, q], dim=1) if q_ctx is not None else q
    k = torch.cat([k_ctx, k], dim=1) if k_ctx is not None else k
    v = torch.cat([v_ctx, v], dim=1) if v_ctx is not None else v
    if image_rotary_emb is not None:
        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)


class Flux2JointAttention(nn.Module):
    def __init__(self, dim, heads=48, head_dim=128, eps=1e-6):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.Dropout(0.0))
        self.add_q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.add_k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.add_v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_add_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, ctx, image_rotary_emb=None):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k = self.norm_k(self.to_k(x).unflatten(-1, (self.heads, -1)))
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        q_ctx = self.norm_added_q(self.add_q_proj(ctx).unflatten(-1, (self.heads, -1)))
        k_ctx = self.norm_added_k(self.add_k_proj(ctx).unflatten(-1, (self.heads, -1)))
        v_ctx = self.add_v_proj(ctx).unflatten(-1, (self.heads, -1))
        out = flux2_attention(q, k, v, q_ctx, k_ctx, v_ctx, image_rotary_emb)
        out = out.flatten(-2).to(q.dtype)
        txt_len = ctx.shape[1]
        ctx_out = self.to_add_out(out[:, :txt_len])
        img_out = self.to_out(out[:, txt_len:])
        return img_out, ctx_out


# --- Fused parallel self-attention + MLP for single-stream (transformer_flux2.py:568-621, 708-783) ---

class Flux2ParallelSelfAttention(nn.Module):
    def __init__(self, dim, heads=48, head_dim=128, mlp_ratio=3.0, mlp_mult_factor=2, eps=1e-6):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.inner_dim = inner_dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor
        self.to_qkv_mlp_proj = nn.Linear(dim, inner_dim * 3 + self.mlp_hidden_dim * mlp_mult_factor, bias=False)
        self.mlp_act_fn = Flux2SwiGLU()
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Linear(inner_dim + self.mlp_hidden_dim, dim, bias=False)

    def forward(self, x, image_rotary_emb=None):
        projected = self.to_qkv_mlp_proj(x)
        qkv, mlp_in = torch.split(projected, [3 * self.inner_dim, self.mlp_hidden_dim * self.mlp_mult_factor], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.norm_q(q.unflatten(-1, (self.heads, -1)))
        k = self.norm_k(k.unflatten(-1, (self.heads, -1)))
        v = v.unflatten(-1, (self.heads, -1))
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(-2).to(q.dtype)
        mlp_out = self.mlp_act_fn(mlp_in)
        return self.to_out(torch.cat([attn_out, mlp_out], dim=-1))


# --- Transformer Blocks (transformer_flux2.py:786-947) ---

class Flux2TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, mlp_ratio=3.0, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2JointAttention(dim, heads=num_heads, head_dim=head_dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim, dim, mult=mlp_ratio)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim, dim, mult=mlp_ratio)

    def forward(self, hidden_states, encoder_hidden_states, temb_mod_img, temb_mod_txt, image_rotary_emb=None):
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(temb_mod_txt, 2)

        norm_x = (1 + scale_msa) * self.norm1(hidden_states) + shift_msa
        norm_ctx = (1 + c_scale_msa) * self.norm1_context(encoder_hidden_states) + c_shift_msa

        attn_out, ctx_attn_out = self.attn(norm_x, norm_ctx, image_rotary_emb)

        hidden_states = hidden_states + gate_msa * attn_out
        norm_ff = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_ff)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * ctx_attn_out
        norm_ctx_ff = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_ctx_ff)

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, mlp_ratio=3.0, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2ParallelSelfAttention(dim, heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, eps=eps)

    def forward(self, hidden_states, temb_mod, image_rotary_emb=None):
        (mod_shift, mod_scale, mod_gate), = Flux2Modulation.split(temb_mod, 1)
        norm_x = (1 + mod_scale) * self.norm(hidden_states) + mod_shift
        return hidden_states + mod_gate * self.attn(norm_x, image_rotary_emb)


# --- Main Model (transformer_flux2.py:1040-1382) ---

class Flux2Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels=128,
        num_layers=8,
        num_single_layers=48,
        attention_head_dim=128,
        num_attention_heads=48,
        joint_attention_dim=15360,
        mlp_ratio=3.0,
        axes_dims_rope=(32, 32, 32, 32),
        rope_theta=2000,
        eps=1e-6,
        guidance_embeds=True,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.out_channels = in_channels

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepEmbedding(inner_dim, guidance_embeds)

        self.double_stream_modulation_img = Flux2Modulation(inner_dim, mod_param_sets=2)
        self.double_stream_modulation_txt = Flux2Modulation(inner_dim, mod_param_sets=2)
        self.single_stream_modulation = Flux2Modulation(inner_dim, mod_param_sets=1)

        self.x_embedder = nn.Linear(in_channels, inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList([
            Flux2TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps)
            for _ in range(num_layers)
        ])
        self.single_transformer_blocks = nn.ModuleList([
            Flux2SingleTransformerBlock(inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps)
            for _ in range(num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=eps)
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
        guidance=None,
    ):
        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)

        double_mod_img = self.double_stream_modulation_img(temb)
        double_mod_txt = self.double_stream_modulation_txt(temb)
        single_mod = self.single_stream_modulation(temb)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, double_mod_img, double_mod_txt, concat_rotary_emb,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, single_mod, concat_rotary_emb)

        hidden_states = hidden_states[:, num_txt_tokens:]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


if __name__ == "__main__":
    print("Flux2 Transformer Model")
    print("=" * 40)
    model = Flux2Transformer2DModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e9:.2f}B")
    print(f"\nArchitecture: {model.inner_dim}d, "
          f"{len(model.transformer_blocks)} double + {len(model.single_transformer_blocks)} single blocks")
    print("\nKey differences from FLUX.1:")
    print("- Shared modulation (3 Flux2Modulation heads at model level)")
    print("- SwiGLU activation (not GELU-approximate)")
    print("- Fused QKV+MLP parallel single-stream blocks")
    print("- No pooled_projections, guidance always on")
    print("- RoPE: theta=2000, axes=(32,32,32,32), separate text/image embeddings")
