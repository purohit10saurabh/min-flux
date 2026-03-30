# FLUX.1 Latent Utilities Documentation

## Overview

This document describes `utils/latent.py`: latent packing and unpacking plus 3D position IDs for image latent tokens in FLUX.1. Packed latents use a 2×2 spatial grouping: each patch merges four spatial positions into the channel dimension, then patches are flattened into the sequence dimension `(B, H/2·W/2, C·4)`. Position IDs tag each latent token with `(axis0, row, col)` where axis 0 stays zero for the image stream.

---

## Source of Truth

### Canonical Source Files

| Short Name | Full Path |
|------------|-----------|
| `pipeline_flux` | [`src/diffusers/pipelines/flux/pipeline_flux.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/pipelines/flux/pipeline_flux.py) |

### Line-by-Line Mapping

| minFLUX function / block | Canonical Source | Source Lines | Verdict |
|----------------------------|------------------|--------------|---------|
| `prepare_latent_image_ids` | `FluxPipeline._prepare_latent_image_ids` | 506-518 | EXACT MATCH (intermediate `shape` unpack inlined into `reshape(height * width, 3)`; `batch_size` unused in both signatures) |
| `pack_latents` | `FluxPipeline._pack_latents` | 520-526 | EXACT MATCH |
| `unpack_latents` | `FluxPipeline._unpack_latents` | 528-542 | EXACT MATCH (`channels // 4` equals `channels // (2 * 2)` in the reference) |

### Notes

- **`prepare_latent_image_ids`**: Channel 0 is left at zero; channels 1 and 2 receive row and column indices. The result has shape `(H·W, 3)` before `.to(device, dtype)`.
- **`pack_latents`**: `view` splits `H` and `W` into `(H/2, 2)` and `(W/2, 2)`; `permute` orders batch, patch grid, channels, and the 2×2 within each patch; `reshape` produces sequence length `(H/2)*(W/2)` and feature dim `C*4`.
- **`unpack_latents`**: Latent spatial size is derived from pixel `height`/`width` and `vae_scale_factor` so packed geometry stays consistent with VAE compression and the divisibility-by-2 requirement for packing.
