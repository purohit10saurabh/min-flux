# Shared Model Building Blocks Documentation

## Overview

`utils/model.py` contains all model-level building blocks shared by both FLUX.1 and FLUX.2: sinusoidal timestep embeddings, RoPE positional embeddings (1D frequency tables + application + multi-axis module), adaptive layer normalization, and the joint attention function.

---

## Source of Truth

### Canonical Source Files


| Short Name          | Full Path                                                                                                                                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embeddings`        | `[src/diffusers/models/embeddings.py](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/embeddings.py)`                                         |
| `normalization`     | `[src/diffusers/models/normalization.py](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/normalization.py)`                                   |
| `transformer_flux`  | `[src/diffusers/models/transformers/transformer_flux.py](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/transformers/transformer_flux.py)`   |
| `transformer_flux2` | `[src/diffusers/models/transformers/transformer_flux2.py](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/transformers/transformer_flux2.py)` |


### Line-by-Line Mapping


| minFLUX symbol            | Canonical Source                                                                                | Source Lines      | Verdict                                                           |
| ------------------------- | ----------------------------------------------------------------------------------------------- | ----------------- | ----------------------------------------------------------------- |
| `get_timestep_embedding`  | `embeddings.get_timestep_embedding`                                                             | 26-77             | EXACT MATCH                                                       |
| `TimestepEmbedding`       | `embeddings.TimestepEmbedding`                                                                  | 1261-1306         | SIMPLIFIED (keeps only `in_channels`, `out_channels`, `bias`)     |
| `get_1d_rotary_pos_embed` | `embeddings.get_1d_rotary_pos_embed`                                                            | 1119-1183         | MATCH (Flux path: `use_real=True`, `repeat_interleave_real=True`) |
| `apply_rotary_emb`        | `embeddings.apply_rotary_emb`                                                                   | 1186-1240         | MATCH (Flux path: `use_real=True`, `unbind_dim=-1`)               |
| `PosEmbed`                | `transformer_flux.FluxPosEmbed` / `transformer_flux2.Flux2PosEmbed`                             | 494-522 / 950-979 | MATCH (callers pass `theta`/`axes_dim`)                           |
| `AdaLayerNormContinuous`  | `normalization.AdaLayerNormContinuous`                                                          | 307-351           | MATCH (`bias` param: FLUX.1 uses `True`, FLUX.2 uses `False`)     |
| `joint_attention`         | `transformer_flux.FluxAttnProcessor.__call__` / `transformer_flux2.Flux2AttnProcessor.__call__` | 75-140 / 325-391  | MATCH (inlined as `F.scaled_dot_product_attention`)               |


### Notes

- `**PosEmbed**`: Callers specify `theta` and `axes_dim` (FLUX.1: theta=10000, 3 axes; FLUX.2: theta=2000, 4 axes).
- `**AdaLayerNormContinuous**`: FLUX.1 uses default `bias=True`; FLUX.2 passes `bias=False`.
- `**joint_attention**`: Concatenates context Q/K/V with image Q/K/V, applies RoPE, runs SDPA, returns the combined output.

