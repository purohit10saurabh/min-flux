# Flux2 Inference Loop Documentation

## Overview

FLUX.2 inference follows the same Euler ODE sampling as FLUX.1 but differs in timestep shifting, VAE decoding, and transformer interface.

## Key Differences from FLUX.1

### `compute_empirical_mu` (replaces `calculate_shift`)

FLUX.2 uses an empirically fitted piecewise-linear function of both image sequence length and number of steps, rather than FLUX.1's resolution-only linear shift.

For large images (`seq_len > 4300`): `mu = a2 * seq_len + b2`
Otherwise: linear interpolation between 10-step and 200-step fitted lines.

### VAE Decode (BatchNorm de-normalization)

FLUX.1: `latents = (latents / scaling_factor) + shift_factor`
FLUX.2: `latents = latents * bn_std + bn_mean` then `unpatchify_latents`

The BatchNorm running statistics come from `vae.bn` (initialized in `AutoencoderKLFlux2.__init__`).

### No `pooled_projections`

FLUX.2's transformer does not take pooled text embeddings.

---

## Source of Truth

### Canonical Source Files

| Short Name | Full Path |
|------------|-----------|
| `pipeline_flux2` | `diffusers/src/diffusers/pipelines/flux2/pipeline_flux2.py` |
| `scheduler` | `diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py` |
| `transformer_flux2` | `diffusers/src/diffusers/models/transformers/transformer_flux2.py` |

### Line-by-Line Mapping

| min-flux function / block | Canonical Source | Source Lines | Verdict |
|---------------------------|------------------|--------------|---------|
| `compute_empirical_mu` | `pipeline_flux2.compute_empirical_mu` | 159-174 | EXACT MATCH |
| `get_sigmas_flux2` (linspace + exp(mu) shift) | BFL `sampling.generalized_time_snr_shift` + `sampling.get_schedule` | 240-248 | MATCH (inlined, rearranged) |
| `euler_step` | `scheduler.step` | 507-508 | IMPORTED from flux1/inference_loop |
| Latent preparation (randn + pack) | `pipeline_flux2.prepare_latents` | 619-650 | MATCH (simplified) |
| `prepare_latent_ids` | `pipeline_flux2._prepare_latent_ids` | 375-404 | IMPORTED from flux2/training_loop |
| Transformer forward kwargs (no pooled_projections) | `pipeline_flux2.__call__` | 971-980 | EXACT MATCH |
| Unpack + BN denorm + unpatchify + VAE decode | `pipeline_flux2.__call__` | 1014-1024 | EXACT MATCH |

### Notes

- **`compute_empirical_mu`**: The coefficients `a1, b1, a2, b2` are empirically fitted constants from the FLUX.2 release, not derived from theory.
- **Sigma shift**: The BFL reference uses `generalized_time_snr_shift(t, mu, 1.0) = exp(mu) / (exp(mu) + (1/t - 1))`. We inline this as `shift * t / (1 + (shift - 1) * t)` where `shift = exp(mu)`.
- **Unpack**: Training uses the simple `unpack_latents` (permute+reshape). Inference here does the same inline since the pipeline's `_unpack_latents_with_ids` (position-ID scatter) is equivalent for fixed-resolution single-image inference.
