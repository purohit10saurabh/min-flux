"""
Minimal Flux (FLUX.1) training — the complete training algorithm.
Uses the minimal transformer (flux1/model.py) and VAE (flux1/vae.py) from this repo.

References (source of truth):
1) BFL flux-inference — AutoEncoder encode, scale/shift normalization:
   https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
2) diffusers training utilities — timestep density sampling, loss weighting:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
3) diffusers FluxPipeline — _pack_latents, _unpack_latents, _prepare_latent_image_ids:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
4) SD3 paper (Esser et al., 2024) — rectified flow, weighting schemes:
   https://arxiv.org/abs/2403.03206
"""

import torch

from utils.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    get_sigmas,
)
from utils.latent import prepare_latent_image_ids, pack_latents, unpack_latents


def flux_training_step(
    transformer,
    vae,
    optimizer,
    lr_scheduler,
    pixel_values: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    accelerator,
    weight_dtype: torch.dtype,
    weighting_scheme: str = "none",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    guidance_scale: float = 3.5,
    max_grad_norm: float = 1.0,
    num_train_timesteps: int = 1000,
):
    model_input = vae.encode(pixel_values).to(dtype=weight_dtype)

    latent_image_ids = prepare_latent_image_ids(
        model_input.shape[0], model_input.shape[2] // 2, model_input.shape[3] // 2,
        accelerator.device, weight_dtype,
    )

    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme, batch_size=bsz,
        logit_mean=logit_mean, logit_std=logit_std, mode_scale=mode_scale,
    )
    indices = (u * num_train_timesteps).long()
    sigmas = get_sigmas(indices, num_train_timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
    timesteps = sigmas.flatten() * num_train_timesteps
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

    packed_noisy_model_input = pack_latents(
        noisy_model_input, batch_size=bsz, num_channels_latents=model_input.shape[1],
        height=model_input.shape[2], width=model_input.shape[3],
    )

    guidance = None
    if transformer.guidance_embeds:
        guidance = torch.tensor([guidance_scale], device=accelerator.device).expand(bsz)

    model_pred = transformer(
        hidden_states=packed_noisy_model_input, timestep=timesteps / 1000, guidance=guidance,
        pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids, img_ids=latent_image_ids,
    )

    model_pred = unpack_latents(
        model_pred, height=model_input.shape[2] * vae.vae_scale_factor,
        width=model_input.shape[3] * vae.vae_scale_factor, vae_scale_factor=vae.vae_scale_factor,
    )

    weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
    target = noise - model_input
    loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
    loss = loss.mean()

    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(transformer.parameters(), max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return loss.detach().item()


def flux_training(
    transformer, vae, optimizer, lr_scheduler,
    train_dataloader, accelerator, num_epochs: int,
    weight_dtype: torch.dtype = torch.bfloat16, weighting_scheme: str = "none",
    guidance_scale: float = 3.5, max_grad_norm: float = 1.0,
):
    from tqdm import tqdm

    global_step = 0
    for epoch in range(num_epochs):
        transformer.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)

        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                loss = flux_training_step(
                    transformer=transformer, vae=vae,
                    optimizer=optimizer, lr_scheduler=lr_scheduler,
                    pixel_values=batch["pixel_values"], prompt_embeds=batch["prompt_embeds"],
                    pooled_prompt_embeds=batch["pooled_prompt_embeds"], text_ids=batch["text_ids"],
                    accelerator=accelerator, weight_dtype=weight_dtype,
                    weighting_scheme=weighting_scheme, guidance_scale=guidance_scale, max_grad_norm=max_grad_norm,
                )

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            progress_bar.set_postfix(loss=loss)

        progress_bar.close()

    return global_step
