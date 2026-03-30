# Flux Kontext Training Documentation

## Overview

FLUX.1-Kontext is a reference-image conditioned variant of FLUX.1. The architecture is identical — the same `FluxTransformer2DModel` — but the input sequence includes both noise tokens (to denoise) and reference image tokens (for conditioning).

## What Changes vs Base Flux Training

Only 5 additions to the base training step:

### 1. Encode Reference Image

```python
ref_latents = vae.encode(reference_pixel_values, sample=False)
```

With `sample=False`, encoding is deterministic — the reference should be a fixed conditioning signal.

### 2. Reference Position IDs

```python
ref_ids = prepare_latent_image_ids(...)
ref_ids[..., 0] = 1  # first dim = 1 to distinguish from noise (which has 0)
```

The first coordinate axis differentiates noise tokens (0) from reference tokens (1).

### 3. Concatenate Sequences

```python
hidden_states = cat([packed_noise, packed_ref], dim=1)
img_ids = cat([noise_ids, ref_ids], dim=0)
```

### 4. Slice Output

```python
model_pred = model_pred[:, :packed_noise.shape[1]]
```

Discard the reference token predictions — we only care about the noise prediction.

### 5. Batch Interface

The dataloader must provide `batch["reference_pixel_values"]` in addition to the standard keys.

---

## Source of Truth

### Canonical Source Files

| Short Name | Full Path |
|------------|-----------|
| `kontext_pipeline` | [`src/diffusers/pipelines/flux/pipeline_flux_kontext.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/pipelines/flux/pipeline_flux_kontext.py) |
| `pipeline_flux` | [`src/diffusers/pipelines/flux/pipeline_flux.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/pipelines/flux/pipeline_flux.py) |

### Line-by-Line Mapping

| minFLUX block | Canonical Source | Source Lines | Verdict |
|----------------|------------------|--------------|---------|
| Reference VAE encode (sample=False) | BFL `autoencoder.AutoEncoder.encode` | 308-311 | MATCH (sample flag replaces DiagonalGaussian mode) |
| Reference position IDs (`[..., 0] = 1`) | `kontext_pipeline.prepare_latents` | 715-720 | EXACT MATCH |
| Concatenate hidden_states | `kontext_pipeline.__call__` | 1083-1084 | EXACT MATCH |
| Concatenate img_ids | `kontext_pipeline.__call__` | 1007-1009 | EXACT MATCH |
| Slice output | `kontext_pipeline.__call__` | 1098 | EXACT MATCH |
| Transformer forward kwargs | `kontext_pipeline.__call__` | 1088-1097 | EXACT MATCH |
| All other training logic | `utils/training_utils.py` (imported) | — | INHERITED |
