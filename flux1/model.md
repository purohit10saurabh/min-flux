# Flux Transformer Model Documentation

## Overview

The complete FLUX.1 transformer architecture in one file (~270 lines).

This file defines `FluxJointAttention + FluxTransformerBlock + FluxTransformer2DModel` using `nn.Linear`, `nn.LayerNorm`, `nn.RMSNorm`, and `F.scaled_dot_product_attention`.

## Architecture

```
Input: hidden_states (B, seq, 64) + encoder_hidden_states (B, txt_seq, 4096)
  |
  x_embedder: Linear(64, 3072)           context_embedder: Linear(4096, 3072)
  |                                       |
  +-- 19x FluxTransformerBlock (double-stream) --+
  |   norm1 -> joint_attn -> residual            |
  |   norm2 -> ff -> residual                    |
  |   (text and image processed in parallel)     |
  +----------------------------------------------+
  |
  +-- 38x FluxSingleTransformerBlock (single-stream) --+
  |   cat(text, image) -> norm -> attn+mlp parallel    |
  |   -> gate -> residual -> split(text, image)        |
  +----------------------------------------------------+
  |
  norm_out (AdaLayerNormContinuous) -> proj_out: Linear(3072, 64)
  |
  Output: (B, seq, 64)
```

### Key Design Choices

- **Double-stream blocks** (FluxTransformerBlock): Text and image have separate attention norms, separate FFN paths, but share a single joint attention computation. This allows cross-modal attention while maintaining modality-specific processing.
- **Single-stream blocks** (FluxSingleTransformerBlock): Text and image tokens are concatenated and processed together. Attention and MLP run in parallel (not sequential) and are projected out together.
- **AdaLN-Zero modulation**: Timestep + guidance + text embeddings modulate every block via shift/scale/gate parameters.
- **RoPE**: Rotary positional embeddings applied per-axis (16+56+56 = 128 = head_dim) to queries and keys.

### Default Config (FLUX.1-dev)

| Parameter | Value |
|-----------|-------|
| `in_channels` | 64 |
| `num_layers` (double-stream) | 19 |
| `num_single_layers` | 38 |
| `num_attention_heads` | 24 |
| `attention_head_dim` | 128 |
| `inner_dim` | 3072 |
| `joint_attention_dim` | 4096 |
| `pooled_projection_dim` | 768 |
| `axes_dims_rope` | (16, 56, 56) |
| Parameters | ~12B |

---

## Source of Truth

### Canonical Source Files

| Short Name | Full Path |
|------------|-----------|
| `transformer_flux` | `diffusers/src/diffusers/models/transformers/transformer_flux.py` |
| `embeddings` | `diffusers/src/diffusers/models/embeddings.py` |
| `normalization` | `diffusers/src/diffusers/models/normalization.py` |
| `attention` | `diffusers/src/diffusers/models/attention.py` |
| `activations` | `diffusers/src/diffusers/models/activations.py` |

### Line-by-Line Mapping

| minFLUX class | Canonical Source | Source Lines | Verdict |
|----------------|------------------|--------------|---------|
| `get_timestep_embedding` | `embeddings.get_timestep_embedding` | 26-77 | MATCH (simplified, flip_sin_to_cos=True hardcoded) |
| `TimestepEmbedding` | `embeddings.TimestepEmbedding` | 1261-1306 | MATCH (stripped cond_proj, post_act) |
| `TextProjection` | `embeddings.PixArtAlphaTextProjection` | 2191-2217 | MATCH (act_fn="silu" hardcoded) |
| `FluxTimestepEmbedding` | `embeddings.CombinedTimestepGuidanceTextProjEmbeddings` | 1603-1624 | MATCH (merged guidance/no-guidance variants) |
| `AdaLayerNormZero` | `normalization.AdaLayerNormZero` | 130-170 | MATCH (stripped num_embeddings, norm_type options) |
| `AdaLayerNormZeroSingle` | `normalization.AdaLayerNormZeroSingle` | 173-202 | EXACT MATCH (logic) |
| `AdaLayerNormContinuous` | `normalization.AdaLayerNormContinuous` | 307-351 | MATCH (layer_norm only) |
| `FluxPosEmbed` | `transformer_flux.FluxPosEmbed` | 494-522 | EXACT MATCH (minus MPS/NPU dtype workaround) |
| `flux_attention` | `transformer_flux.FluxAttnProcessor.__call__` | 75-140 | MATCH (inlined, uses F.scaled_dot_product_attention) |
| `FluxJointAttention` | `transformer_flux.FluxAttention` (double-stream) | 275-352 | MATCH (stripped processor dispatch, IP-adapter) |
| `FluxSingleAttention` | `transformer_flux.FluxAttention` (pre_only=True) | 275-352 | MATCH (pre_only variant) |
| `FluxTransformerBlock` | `transformer_flux.FluxTransformerBlock` | 409-491 | MATCH (stripped ControlNet residual) |
| `FluxSingleTransformerBlock` | `transformer_flux.FluxSingleTransformerBlock` | 355-406 | EXACT MATCH (logic) |
| `FluxTransformer2DModel.__init__` | `transformer_flux.FluxTransformer2DModel.__init__` | 579-635 | MATCH (stripped patch_size, out_channels, grad ckpt) |
| `FluxTransformer2DModel.forward` | `transformer_flux.FluxTransformer2DModel.forward` | 637-778 | MATCH (stripped ControlNet, IP-adapter, grad ckpt) |
| FeedForward (inline `nn.Sequential`) | `attention.FeedForward` with `gelu-approximate` | 1696-1742 | MATCH (inlined as Linear->GELU->Linear) |

### What Was Stripped

- **ControlNet residual hooks** (`controlnet_block_samples`, `controlnet_single_block_samples`)
- **IP-Adapter projection** (`encoder_hid_proj`, `ip_adapter_image_embeds`)
- **Gradient checkpointing** (`_gradient_checkpointing_func`)
- **Attention processor dispatch** (FluxAttnProcessor pattern replaced with direct `F.scaled_dot_product_attention`)
- **ConfigMixin / ModelMixin / PeftAdapterMixin** (diffusers infrastructure)
- **fp16 overflow clipping** (`clip(-65504, 65504)`)
- **Fused QKV projections** (`fused_projections` path)
