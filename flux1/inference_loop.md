# Flux Inference Loop Documentation

## Overview

This document explains the FLUX.1 inference (sampling) loop — the reverse process that generates images from noise.

## Key Concepts

### Euler ODE Step (Flow Matching Sampling)

The entire `FlowMatchEulerDiscreteScheduler.step()` distills to one line:

```python
prev_sample = sample + (sigma_next - sigma) * model_output
```

This is the Euler method applied to the ODE `dx/dt = v(x, t)` where `v` is the velocity predicted by the transformer. The step size `dt = sigma_next - sigma` is negative (sigma decreases from 1 to 0), so we move from noise toward data.

### Resolution-Dependent Timestep Shift

`calculate_shift` adjusts the sigma schedule based on image resolution. Higher resolutions need more denoising time at high noise levels. This is a linear interpolation between `base_shift` (for 256-token sequences) and `max_shift` (for 4096-token sequences).

### Sigma Schedule

The schedule is `linspace(1.0, 1/N, N)` shifted by `exp(mu)`:

```
shift = exp(mu)
sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
```

This is the BFL reference formula `exp(mu) / (exp(mu) + (1/t - 1))` (from `https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py`), rearranged into ratio form. A terminal `0.0` is appended for the final step.

---

## Source of Truth

### Canonical Source Files

| Short Name | Full Path |
|------------|-----------|
| `pipeline_flux` | `diffusers/src/diffusers/pipelines/flux/pipeline_flux.py` |
| `scheduler` | `diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py` |
| `transformer_flux` | `diffusers/src/diffusers/models/transformers/transformer_flux.py` |

### Line-by-Line Mapping

| min-flux function / block | Canonical Source | Source Lines | Verdict |
|---------------------------|------------------|--------------|---------|
| `calculate_shift` | `pipeline_flux.calculate_shift` | 74-84 | EXACT MATCH |
| `get_sigmas` (linspace + exp(mu) shift + append 0) | BFL `sampling.time_shift` + `sampling.get_schedule` | 277-305 | MATCH (inlined, rearranged) |
| `euler_step` | `scheduler.step` | 507-508 | EXACT MATCH (Euler ODE step distilled to one line) |
| Latent preparation (randn + pack) | `pipeline_flux.prepare_latents` | 544-597 | MATCH (simplified) |
| `_prepare_latent_image_ids` call | `pipeline_flux._prepare_latent_image_ids` | 506-518 | EXACT MATCH |
| Transformer forward kwargs | `pipeline_flux.__call__` denoise loop | 949-961 | EXACT MATCH (minus cache_context) |
| `timestep / 1000` convention | `transformer_flux.forward` | 688 | CORRECT (transformer does `* 1000` internally) |
| Unpack + inverse normalize + VAE decode | `pipeline_flux.__call__` | 1009-1012 | EXACT MATCH |

### Notes

- **`euler_step`**: The full `scheduler.step()` (lines 425-524) handles per-token timesteps, stochastic sampling, upcasting, and step index tracking. We distill it to the deterministic core: `sample + dt * model_output`.
- **Sigma shift**: The BFL reference uses `time_shift(mu, 1.0, t) = exp(mu) / (exp(mu) + (1/t - 1))`. We inline this as `shift * t / (1 + (shift - 1) * t)` where `shift = exp(mu)`.
- **No `cache_context`**: Diffusers wraps the transformer call in `cache_context("cond")` for compilation optimization. Omitted for minimality.
