"""
Minimal Flux (FLUX.1) training — the complete training algorithm.

References (source of truth):
1) diffusers training utilities — compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
2) diffusers dreambooth LoRA Flux training example — training step structure, get_sigmas:
   https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py
3) diffusers FluxPipeline — _pack_latents, _unpack_latents, _prepare_latent_image_ids:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
4) diffusers FluxTransformer2DModel — forward() signature and timestep*1000 convention:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
5) SD3 paper (Esser et al., 2024) — rectified flow, weighting schemes:
   https://arxiv.org/abs/2403.03206
"""

import torch

from shared.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    get_sigmas,
)
from shared.latent_utils import prepare_latent_image_ids, pack_latents, unpack_latents


def flux_training_step(
    transformer,
    vae,
    noise_scheduler,
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
):
    model_input = vae.encode(pixel_values.to(dtype=vae.dtype)).latent_dist.sample()
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
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
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

    sigmas = get_sigmas(timesteps, noise_scheduler, accelerator.device, n_dim=model_input.ndim, dtype=model_input.dtype)
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

    packed_noisy_model_input = pack_latents(
        noisy_model_input, batch_size=bsz, num_channels_latents=model_input.shape[1],
        height=model_input.shape[2], width=model_input.shape[3],
    )

    guidance = None
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([guidance_scale], device=accelerator.device).expand(bsz)

    model_pred = transformer(
        hidden_states=packed_noisy_model_input, timestep=timesteps / 1000, guidance=guidance,
        pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids, img_ids=latent_image_ids, return_dict=False,
    )[0]

    model_pred = unpack_latents(
        model_pred, height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor, vae_scale_factor=vae_scale_factor,
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
    transformer, vae, noise_scheduler, optimizer, lr_scheduler,
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
                    transformer=transformer, vae=vae, noise_scheduler=noise_scheduler,
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
