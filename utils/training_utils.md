# Shared Training Utilities Documentation

## Overview

`utils/training_utils.py` provides FLUX.1 and FLUX.2 training helpers: timestep density sampling (`compute_density_for_timestep_sampling`), SD3-style loss weights (`compute_loss_weighting_for_sd3`), and flow-matching sigmas (`get_sigmas`). Density sampling supports `uniform`, `logit_normal`, and `mode`. Loss weighting supports `uniform`, `sigma_sqrt`, and `cosmap`. `get_sigmas` replaces consulting a `FlowMatchEulerDiscreteScheduler` table: for default training (fixed `num_train_timesteps`, `shift=1.0`, no dynamic shifting), scheduler sigmas satisfy `sigma[i] = 1 - i / N` with `N = num_train_timesteps` (typically 1000), matching `timesteps = linspace(1, N, N)[::-1]` and `sigmas = timesteps / N` in `FlowMatchEulerDiscreteScheduler.__init__`.

## Source of Truth

### Canonical Source Files


| Short Name             | Full Path                                                                    |
| ---------------------- | ---------------------------------------------------------------------------- |
| `training_utils`       | [`src/diffusers/training_utils.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/training_utils.py) |
| `flow_match_scheduler` | [`src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py) |


### Line-by-Line Mapping


| minFLUX symbol                          | Lines | Canonical Source                                                                              | Source Lines | Verdict                                                                                                                |
| --------------------------------------- | ----- | --------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `compute_density_for_timestep_sampling` | 23-40 | `training_utils.compute_density_for_timestep_sampling`                                        | 360-384      | MATCH (explicit defaults `logit_mean=0.0`, `logit_std=1.0`, `mode_scale=1.29` vs `None` in diffusers; logic identical) |
| `compute_loss_weighting_for_sd3`        | 43-51 | `training_utils.compute_loss_weighting_for_sd3`                                               | 387-402      | EXACT MATCH                                                                                                            |
| `get_sigmas`                            | 54-58 | `flow_match_scheduler.FlowMatchEulerDiscreteScheduler.__init__` (timestep/sigma construction) | 126-131      | MATCH (closed form; see Notes)                                                                                         |


### Notes

- `**get_sigmas` vs scheduler**: Older training code took discretized timestep indices, a `noise_scheduler`, and read `scheduler.sigmas[index]` (or equivalent) after building the scheduler. Here, callers pass tensor `indices` and `num_train_timesteps`; sigmas are computed as `(1.0 - indices.float() / num_train_timesteps)` and unsqueezed to `n_dim` for broadcasting. With `shift=1.0` and `use_dynamic_shifting=False`, `sigmas = timesteps / N` from lines 126-129 is unchanged by the shift formula on lines 130-132, so the values align with the scheduler’s CPU sigma table for those settings.
- **Derivation**: `timesteps = [N, N-1, …, 1]` gives `sigma_k = (N-k)/N` at zero-based position `k`, i.e. `sigma = 1 - i/N` when `i` is the training index associated with that entry.
- **Callers**: FLUX training scripts in this repo should pass the same `indices` / `num_train_timesteps` convention they used when indexing the scheduler, so noisy latents and weighting see consistent `sigma`.

