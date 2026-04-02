"""
Shared training and inference utilities for FLUX.1 and FLUX.2.

Contains: flow-matching noise sampling, velocity loss + optimizer step,
Euler ODE step, and the shared training loop.

References (source of truth):
1) diffusers training utilities — compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
2) diffusers FlowMatchEulerDiscreteScheduler — timestep/sigma tables:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
3) BFL flux — denoise (Euler step):
   https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py
4) SD3 paper (Esser et al., 2024) — rectified flow, weighting schemes:
   https://arxiv.org/abs/2403.03206
"""

import math
import torch
from typing import Optional, Union


def _compute_density_for_timestep_sampling(
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


def _compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def _get_sigmas(indices, num_train_timesteps=1000, n_dim=4, dtype=torch.float32):
    sigma = (1.0 - indices.float() / num_train_timesteps).to(dtype=dtype)
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def sample_flow_match_noise(model_input, weighting_scheme="none", logit_mean=0.0,
                            logit_std=1.0, mode_scale=1.29, num_train_timesteps=1000):
    noise = torch.randn_like(model_input)
    u = _compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme, batch_size=model_input.shape[0],
        logit_mean=logit_mean, logit_std=logit_std, mode_scale=mode_scale,
    )
    indices = (u * num_train_timesteps).long()
    sigmas = _get_sigmas(indices, num_train_timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
    timesteps = sigmas.flatten() * num_train_timesteps
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    return noisy_model_input, noise, sigmas, timesteps


def flow_match_loss_step(model_pred, noise, model_input, sigmas, weighting_scheme,
                         accelerator, transformer, optimizer, lr_scheduler, max_grad_norm):
    weighting = _compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
    target = noise - model_input
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1,
    ).mean()
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(transformer.parameters(), max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    return loss.detach().item()


def euler_step(model_output, sigma, sigma_next, sample):
    return sample + (sigma_next - sigma) * model_output


def train_loop(step_fn, transformer, train_dataloader, accelerator, num_epochs):
    from tqdm import tqdm
    global_step = 0
    for epoch in range(num_epochs):
        transformer.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}",
                            disable=not accelerator.is_local_main_process)
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                loss = step_fn(batch)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            progress_bar.set_postfix(loss=loss)
        progress_bar.close()
    return global_step
