# Shared Training and Inference Utilities Documentation

## Overview

`utils/training.py` contains all training and inference utilities shared by FLUX.1 and FLUX.2: flow-matching noise sampling with timestep density, velocity-target loss computation with optimizer step, the Euler ODE step, and the shared training loop.

---

## Source of Truth

### Canonical Source Files

| Short Name | Full Path |
|------------|-----------|
| `training_utils` | [`src/diffusers/training_utils.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/training_utils.py) |
| `flow_match_scheduler` | [`src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py) |
| `flux_sampling` | [`src/flux/sampling.py`](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/sampling.py) |

### Line-by-Line Mapping

| minFLUX symbol | Canonical Source | Source Lines | Verdict |
|----------------|------------------|--------------|---------|
| `_compute_density_for_timestep_sampling` | `training_utils.compute_density_for_timestep_sampling` | 360-384 | MATCH |
| `_compute_loss_weighting_for_sd3` | `training_utils.compute_loss_weighting_for_sd3` | 387-402 | EXACT MATCH |
| `_get_sigmas` | `flow_match_scheduler.__init__` (sigma construction) | 126-131 | MATCH (closed-form `sigma = 1 - i/N`) |
| `sample_flow_match_noise` | Composed from `_compute_density_for_timestep_sampling` + `_get_sigmas` + flow interpolation | — | Wraps the 3 primitives above into one call: returns `(noisy_input, noise, sigmas, timesteps)` |
| `flow_match_loss_step` | Composed from `_compute_loss_weighting_for_sd3` + MSE on velocity target + Accelerate optimizer step | — | Wraps loss + backward + clip + step into one call: returns `loss.item()` |
| `euler_step` | `flux_sampling.denoise` (inner step) | 115-116 | EXACT MATCH (`sample + (sigma_next - sigma) * model_output`) |
| `train_loop` | N/A (minFLUX utility) | — | Shared Accelerate training loop; takes a `step_fn(batch) -> float` closure |
