# Flux2 Training Documentation

## Overview

This document explains FLUX.2 training, how it differs from FLUX.1, and maps every code block to its canonical diffusers source.

## Key Papers

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | Lipman et al. | 2023 (ICLR) | Foundation of flow matching training |
| [Scaling Rectified Flow Transformers (SD3)](https://arxiv.org/abs/2403.03206) | Esser et al. | 2024 | Weighting schemes, architecture |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | Liu et al. | 2022 | Linear interpolation formulation |

---

## FLUX.1 vs FLUX.2 Architecture Comparison

| Aspect | FLUX.1 | FLUX.2 |
|--------|--------|--------|
| **Text encoder** | CLIP + T5-XXL | Mistral3 (`Mistral3ForConditionalGeneration`) |
| **VAE** | `AutoencoderKL` | `AutoencoderKLFlux2` |
| **VAE normalization** | `(x - shift_factor) * scaling_factor` | `(patchify(x) - bn_mean) / bn_std` (BatchNorm) |
| **Latent channels** | 16 (64/4 packed) | 32 (128/4 patchified) |
| **Packing** | 2x2 patch rearrange in one step | `_patchify_latents` then `_pack_latents` (flatten) |
| **Position IDs** | 3D `(ch, H, W)` | 4D `(T, H, W, L)` |
| **Transformer `in_channels`** | 64 | 128 |
| **Double-stream blocks** | 19 | 8 |
| **Single-stream blocks** | 38 | 48 |
| **Attention heads** | 24 × 128 | 48 × 128 |
| **Joint attention dim** | 4096 | 15360 |
| **RoPE axes** | `(16, 56, 56)` | `(32, 32, 32, 32)` |
| **Modulation** | Per-block AdaLayerNorm | Shared across all blocks |
| **`pooled_projections`** | Yes (from CLIP) | No |
| **`guidance_embeds`** | Optional (False for schnell) | Always True |
| **Default `guidance_scale`** | 3.5 | 2.5 |
| **Unpacking** | Structured reshape | Position-ID scatter + unpatchify |

---

## Training Components

### 1. VAE Encoding + BatchNorm Normalization

```python
model_input = vae.encode(pixel_values)
```

**FLUX.1** uses `(latents - shift_factor) * scaling_factor`.

The minimal `Flux2AutoEncoder.encode` handles quant_conv, mean extraction, patchification, and BatchNorm normalization internally.

**FLUX.2** latent layout after encode: patchified 2×2 patches `(B, 128, H/2, W/2)` with BatchNorm applied using the VAE's running statistics. The BatchNorm layer is initialized in `Flux2AutoEncoder.__init__` with `affine=False` and `track_running_stats=True`. The running mean and variance are loaded from the pretrained checkpoint.

### 2. Patchification

`patchify_latents` is no longer a standalone function in `flux2/training.py`; the live implementation is `Flux2AutoEncoder._patchify` in `flux2/vae.py` (lines 52–54).

```python
def patchify_latents(latents):
    # (B, C, H, W) -> (B, C*4, H/2, W/2)
    latents = latents.view(B, C, H//2, 2, W//2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(B, C*4, H//2, W//2)
```

This rearranges each 2×2 spatial patch into the channel dimension. For 32 latent channels, this produces 128 channels at half resolution. This is done **before** the BatchNorm normalization (the BN operates on 128 channels).

FLUX.1 folds this into `_pack_latents` as a single reshape. FLUX.2 separates patchification from sequence packing.

### 3. Sequence Packing

```python
def pack_latents(latents):
    # (B, C, H, W) -> (B, H*W, C)
    return latents.reshape(B, C, H*W).permute(0, 2, 1)
```

FLUX.2's pack is a simple channel-last flatten. FLUX.1's `_pack_latents` does the 2×2 rearrangement AND flattening in one operation.

### 4. Position IDs (4D)

```python
def prepare_latent_ids(latents):
    ids = torch.cartesian_prod(arange(1), arange(H), arange(W), arange(1))
    return ids.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 4)
```

Each image token gets 4D coordinates `(T=0, H=h, W=w, L=0)`. Text tokens get `(T=0, H=0, W=0, L=seq_idx)`.

FLUX.1 uses 3D coordinates `(ch=0, H=h, W=w)` via `_prepare_latent_image_ids`.

The 4th dimension `T` enables multi-image/reference-image support (different T values per image).

### 5. Transformer Forward

```python
model_pred = transformer(
    hidden_states=packed_noisy,
    timestep=timesteps / 1000,
    guidance=guidance,
    encoder_hidden_states=prompt_embeds,
    txt_ids=text_ids,
    img_ids=latent_ids,
)
```

Key differences from FLUX.1:
- **No `pooled_projections`**: FLUX.2 uses `Flux2TimestepGuidanceEmbeddings` which only takes timestep + guidance (no pooled text features). FLUX.1's `CombinedTimestepGuidanceTextProjEmbeddings` also takes pooled CLIP features.
- **Guidance always created**: `guidance_embeds=True` by default in FLUX.2. FLUX.1-schnell has `guidance_embeds=False`.
- **Shared modulation**: FLUX.2 computes modulation once and broadcasts to all blocks. FLUX.1 computes per-block.

### 6. Loss (Shared with FLUX.1)

The flow matching objective is identical:
- **Target**: `noise - model_input` (velocity prediction)
- **Loss**: Weighted MSE
- **Timestep sampling**: `logit_normal`, `mode`, or uniform
- **Loss weighting**: `sigma_sqrt`, `cosmap`, or uniform

These are imported from `utils/training_utils.py`.

---

## Source of Truth

### Canonical Source Files

| Short Name | Full Path |
|------------|-----------|
| `transformer_flux2` | [`src/diffusers/models/transformers/transformer_flux2.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/transformers/transformer_flux2.py) |
| `pipeline_flux2` | [`src/diffusers/pipelines/flux2/pipeline_flux2.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/pipelines/flux2/pipeline_flux2.py) |
| `autoencoder_flux2` | [`src/diffusers/models/autoencoders/autoencoder_kl_flux2.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py) |
| `flux2_ae` | [`src/flux2/autoencoder.py`](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/autoencoder.py) |
| `training` | [`src/diffusers/training.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/training.py) |

### Line-by-Line Mapping

| minFLUX function / block | Lines | Canonical Source | Source Lines | Verdict |
|---------------------------|-------|------------------|--------------|---------|
| `Flux2AutoEncoder._patchify` | flux2/vae.py 52-54 | BFL `autoencoder.AutoEncoder.encode` (rearrange) | 318-324 | MATCH |
| `Flux2AutoEncoder._unpatchify` | flux2/vae.py 56-58 | BFL `autoencoder.AutoEncoder.decode` (rearrange) | 329-334 | MATCH (decode unpatchify) |
| `pack_latents` | 29-30 | `pipeline_flux2._pack_latents` | 473-481 | EXACT MATCH |
| `unpack_latents` | 33-34 | Inverse of `_pack_latents` | N/A | DERIVED (simple permute+reshape inverse) |
| `prepare_latent_ids` | 37-40 | `pipeline_flux2._prepare_latent_ids` | 375-404 | MATCH (simplified, no docstring) |
| VAE encode (quant_conv, mean, patchify, BN) | 49 | BFL `autoencoder.AutoEncoder.encode` | 314-325 | MATCH (minimal VAE uses `quant_conv`; BFL uses encoder moments) |
| Position ID preparation | 50 | `pipeline_flux2.prepare_latents` | 646-647 | EXACT MATCH |
| Timestep sampling + sigmas | 51-52 | `utils/training.compute_density_for_timestep_sampling`, `get_sigmas` (diffusers `training_utils` + FlowMatch sigma table) | 23-40, 54-58 | IMPORTED / inlined (no `noise_scheduler`) |
| Noise interpolation | 51 | Same as FLUX.1 (rectified flow) | N/A | SHARED |
| Pack noisy input | 54 | `pipeline_flux2.prepare_latents` | 649 | EXACT MATCH |
| Guidance (always on) | 55 | `pipeline_flux2.__call__` | 948-949 | EXACT MATCH |
| Transformer forward | 57-60 | `pipeline_flux2.__call__` denoise | 971-980 | EXACT MATCH (no `pooled_projections`) |
| Unpack model prediction | 61 | Inverse of pack (training only) | N/A | DERIVED |
| Loss computation | 63-64 | Same as FLUX.1 (flow matching MSE) | N/A | IMPORTED from utils.training |

### Notes

- **`unpack_latents`**: Not directly from diffusers. The pipeline uses `_unpack_latents_with_ids` (position-ID-based scatter) for variable-resolution support. For fixed-resolution training, the simple inverse of `pack_latents` is mathematically equivalent and avoids the scatter overhead.
- **Encode path**: The minimal `Flux2AutoEncoder.encode` takes the mean latent (no diagonal Gaussian `.sample()`); BFL `AutoEncoder.encode` matches patchify + BatchNorm after the mean split.
- **`guidance_scale=2.5`**: Default from `Flux2Pipeline.__call__`. FLUX.1 uses 3.5.
- **`text_ids`**: Expected shape `(B, seq_len, 4)` with coordinates `(T=0, H=0, W=0, L=token_idx)`. Must be supplied in the batch (see `pipeline_flux2._prepare_text_ids` for reference).

---

## References

1. Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. ICLR 2023. https://arxiv.org/abs/2210.02747

2. Liu, X., Gong, C., & Liu, Q. (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. https://arxiv.org/abs/2209.03003

3. Esser, P., Kulal, S., Blattmann, A., et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. https://arxiv.org/abs/2403.03206

4. Black Forest Labs. (2024). FLUX Model. https://blackforestlabs.ai/

5. Black Forest Labs. (2025). FLUX.2. https://bfl.ai/blog/flux-2
