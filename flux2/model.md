# Flux2 Transformer Model Documentation

## Overview

The FLUX.2 transformer architecture in one file (~350 lines). Same role as `flux1/model.py` for FLUX.1.

## Architecture Differences from FLUX.1


| Aspect             | FLUX.1                                                     | FLUX.2                                                                    |
| ------------------ | ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Modulation**     | Per-block `AdaLayerNormZero` (6 params each)               | 3 shared `Flux2Modulation` heads at model level                           |
| **FFN**            | `Linear -> GELU(tanh) -> Linear`                           | `Linear -> SwiGLU -> Linear` (SiLU-gated)                                 |
| **Single-stream**  | Separate attn + MLP, `proj_out(cat(attn, mlp))`            | Fused `to_qkv_mlp_proj`, parallel attn+MLP, fused `to_out`                |
| **Timestep embed** | `FluxTimestepEmbedding` with pooled text projection        | `Flux2TimestepEmbedding` — timestep + guidance only                       |
| **RoPE theta**     | 10000                                                      | 2000                                                                      |
| **RoPE axes**      | `(16, 56, 56)` — 3 axes, single `pos_embed(cat(txt, img))` | `(32, 32, 32, 32)` — 4 axes, separate `pos_embed(img)` + `pos_embed(txt)` |
| **Biases**         | Most `bias=True`                                           | All `bias=False`                                                          |
| **Default config** | 19 double + 38 single blocks, 24 heads, 3072d                              | 8 double + 48 single blocks, 48 heads, 6144d                                              |


## Architecture

```mermaid
flowchart TD
    subgraph inputs [Inputs]
        ImgIn["hidden_states<br/>(B, seq, 128)"]
        TxtIn["encoder_hidden_states<br/>(B, txt_seq, 15360)"]
        Timestep["timestep + guidance"]
        PosIDs["img_ids + txt_ids<br/>(4D: T, H, W, L)"]
    end

    ImgIn --> XEmb["x_embedder<br/>Linear(128, 6144, bias=False)"]
    TxtIn --> CtxEmb["context_embedder<br/>Linear(15360, 6144, bias=False)"]
    Timestep --> TEmb["Flux2TimestepEmbedding<br/>sinusoidal -> MLP -> t_emb"]
    PosIDs --> RoPE["Flux2PosEmbed<br/>RoPE (32+32+32+32 axes, theta=2000)<br/>separate text / image embeddings"]

    TEmb --> SharedMod["Shared Modulation Heads<br/>double_img / double_txt / single<br/>(computed once, broadcast to all blocks)"]

    subgraph double [" 8x Flux2TransformerBlock (double-stream) "]
        direction TB
        D_Mod["Flux2Modulation.split<br/>shift/scale/gate from shared mod"]
        D_Attn["Flux2JointAttention<br/>Q,K,V from image + text<br/>concat -> RoPE -> SDPA -> split"]
        D_FF["SwiGLU FFN (per stream)<br/>Linear -> SiLU gate -> Linear"]
        D_Mod --> D_Attn --> D_FF
    end

    XEmb -->|"image stream"| double
    CtxEmb -->|"text stream"| double
    SharedMod -.->|"double_img + double_txt"| double
    RoPE -.->|"concat rotary emb"| double

    double --> CatStreams["cat(text, image) along seq"]

    subgraph single [" 48x Flux2SingleTransformerBlock (single-stream) "]
        direction TB
        S_Mod["Flux2Modulation.split<br/>shift/scale/gate from shared mod"]
        S_Par["Flux2ParallelSelfAttention<br/>fused QKV + MLP projection<br/>-> RoPE -> SDPA + SwiGLU in parallel<br/>-> fused to_out"]
        S_Mod --> S_Par
    end

    CatStreams --> single
    SharedMod -.->|"single_mod"| single
    RoPE -.->|"concat rotary emb"| single

    single --> SliceTxt["slice off text tokens"]
    SliceTxt --> NormOut["AdaLayerNormContinuous<br/>conditioned on t_emb"]
    NormOut --> ProjOut["proj_out: Linear(6144, 128, bias=False)"]
    ProjOut --> Output["output (B, seq, 128)"]
```

---

## Source of Truth

### Canonical Source Files


| Short Name          | Full Path                                                          |
| ------------------- | ------------------------------------------------------------------ |
| `transformer_flux2` | [`src/diffusers/models/transformers/transformer_flux2.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/transformers/transformer_flux2.py) |
| `embeddings`        | [`src/diffusers/models/embeddings.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/embeddings.py)                     |
| `normalization`     | [`src/diffusers/models/normalization.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/normalization.py)                  |


### Line-by-Line Mapping


| minFLUX class                      | Canonical Source                                                                  | Source Lines     | Verdict                                         |
| ---------------------------------- | --------------------------------------------------------------------------------- | ---------------- | ----------------------------------------------- |
| `get_timestep_embedding`           | `embeddings.get_timestep_embedding`                                               | 26-77            | MATCH (simplified)                              |
| `TimestepEmbedding`                | `embeddings.TimestepEmbedding`                                                    | 1261-1306        | MATCH (bias=False)                              |
| `Flux2TimestepEmbedding`           | `transformer_flux2.Flux2TimestepGuidanceEmbeddings`                               | 982-1014         | EXACT MATCH                                     |
| `Flux2Modulation`                  | `transformer_flux2.Flux2Modulation`                                               | 1017-1037        | EXACT MATCH (incl. split)                       |
| `AdaLayerNormContinuous`           | `normalization.AdaLayerNormContinuous`                                            | 307-351          | MATCH (bias=False)                              |
| `Flux2PosEmbed`                    | `transformer_flux2.Flux2PosEmbed`                                                 | 950-979          | EXACT MATCH (minus MPS/NPU workaround)          |
| `Flux2SwiGLU`                      | `transformer_flux2.Flux2SwiGLU`                                                   | 283-296          | EXACT MATCH                                     |
| `Flux2FeedForward`                 | `transformer_flux2.Flux2FeedForward`                                              | 299-322          | EXACT MATCH                                     |
| `flux2_attention`                  | `transformer_flux2.Flux2AttnProcessor.__call__`                                   | 325-391          | MATCH (inlined, F.scaled_dot_product_attention) |
| `Flux2JointAttention`              | `transformer_flux2.Flux2Attention`                                                | 493-548          | MATCH (stripped processor dispatch)             |
| `Flux2ParallelSelfAttention`       | `transformer_flux2.Flux2ParallelSelfAttention` + `Flux2ParallelSelfAttnProcessor` | 568-621, 708-783 | MATCH (inlined processor)                       |
| `Flux2TransformerBlock`            | `transformer_flux2.Flux2TransformerBlock`                                         | 855-947          | EXACT MATCH (logic)                             |
| `Flux2SingleTransformerBlock`      | `transformer_flux2.Flux2SingleTransformerBlock`                                   | 786-852          | MATCH (simplified, no split_hidden_states)      |
| `Flux2Transformer2DModel.__init`__ | `transformer_flux2.Flux2Transformer2DModel.__init_`_                              | 1094-1174        | MATCH (stripped KV cache, grad ckpt)            |
| `Flux2Transformer2DModel.forward`  | `transformer_flux2.Flux2Transformer2DModel.forward`                               | 1228-1381        | MATCH (stripped KV cache, grad ckpt)            |


### What Was Stripped

- **KV cache** (`kv_cache`, `kv_cache_mode`, `num_ref_tokens`, `ref_fixed_timestep`) — Klein KV variant
- **Gradient checkpointing** (`_gradient_checkpointing_func`)
- **Attention processor dispatch** (inlined directly as `F.scaled_dot_product_attention`)
- **ConfigMixin / ModelMixin / PeftAdapterMixin** (diffusers infrastructure)
- **fp16 overflow clipping** (`clip(-65504, 65504)`)

