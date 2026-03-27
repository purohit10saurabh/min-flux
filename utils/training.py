"""
Shared training utilities for FLUX.1 and FLUX.2.

References (source of truth):
1) diffusers training utilities — compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
2) diffusers dreambooth LoRA Flux training example — get_sigmas:
   https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py
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


def get_sigmas(timesteps, noise_scheduler, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
