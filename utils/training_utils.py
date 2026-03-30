"""
Shared training utilities for FLUX.1 and FLUX.2.

Replaces diffusers FlowMatchEulerDiscreteScheduler with direct computation.
FlowMatchEulerDiscreteScheduler(num_train_timesteps=N) produces:
  timesteps = [N, N-1, ..., 1], sigmas = timesteps / N = [1.0, ..., 1/N]
  => sigma[i] = 1 - i/N for index i

References (source of truth):
1) diffusers training utilities — compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
2) diffusers FlowMatchEulerDiscreteScheduler — timestep/sigma tables:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
3) SD3 paper (Esser et al., 2024) — rectified flow, weighting schemes:
   https://arxiv.org/abs/2403.03206
"""

import math
import torch
from typing import Optional, Union


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def get_sigmas(indices, num_train_timesteps=1000, n_dim=4, dtype=torch.float32):
    sigma = (1.0 - indices.float() / num_train_timesteps).to(dtype=dtype)
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
