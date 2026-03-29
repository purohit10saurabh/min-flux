"""
Minimal Flux2 (FLUX.2) training — the complete training algorithm.
Uses diffusers model objects (Flux2Transformer2DModel, AutoencoderKLFlux2), not the minimal model classes in this repo.

Key differences from FLUX.1 (flux1/training.py):
- VAE: AutoencoderKLFlux2 with BatchNorm normalization (not shift_factor/scaling_factor)
- Latents: patchified (2x2) then flattened (not 2x2-patch-rearranged in one step)
- Position IDs: 4D (T, H, W, L) not 3D (ch, H, W)
- Transformer: no pooled_projections, guidance always on, shared modulation
- Text encoder: Mistral3 (not CLIP+T5)

References (source of truth):
1) diffusers Flux2Transformer2DModel — forward() signature, shared modulation:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py
2) diffusers Flux2Pipeline — _patchify_latents, _pack_latents, _prepare_latent_ids, _encode_vae_image:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2.py
3) diffusers AutoencoderKLFlux2 — BatchNorm on patchified latents:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py
4) diffusers training utilities — timestep sampling, loss weighting (shared with FLUX.1):
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
5) SD3 paper (Esser et al., 2024) — rectified flow, weighting schemes:
   https://arxiv.org/abs/2403.03206
"""

import torch

from utils.training import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    get_sigmas,
)


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, _, num_channels = latents.shape
    return latents.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = latents.shape
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1))
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = prompt_embeds.shape
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(seq_len))
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def flux2_training_step(
    transformer, vae, noise_scheduler, optimizer, lr_scheduler,
    pixel_values: torch.Tensor, prompt_embeds: torch.Tensor, text_ids: torch.Tensor,
    accelerator, weight_dtype: torch.dtype, weighting_scheme: str = "none",
    logit_mean: float = 0.0, logit_std: float = 1.0, mode_scale: float = 1.29,
    guidance_scale: float = 2.5, max_grad_norm: float = 1.0,
):
    model_input = vae.encode(pixel_values.to(dtype=vae.dtype)).latent_dist.sample()
    model_input = patchify_latents(model_input)

    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(model_input.device, model_input.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(model_input.device, model_input.dtype)
    model_input = (model_input - bn_mean) / bn_std
    model_input = model_input.to(dtype=weight_dtype)

    latent_ids = prepare_latent_ids(model_input).to(device=accelerator.device, dtype=weight_dtype)

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

    packed_noisy_model_input = pack_latents(noisy_model_input)

    guidance = torch.full([1], guidance_scale, device=accelerator.device, dtype=torch.float32).expand(bsz)

    model_pred = transformer(
        hidden_states=packed_noisy_model_input, timestep=timesteps / 1000, guidance=guidance,
        encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_ids, return_dict=False,
    )[0]

    model_pred = unpack_latents(model_pred, model_input.shape[2], model_input.shape[3])

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


def flux2_training(
    transformer, vae, noise_scheduler, optimizer, lr_scheduler,
    train_dataloader, accelerator, num_epochs: int,
    weight_dtype: torch.dtype = torch.bfloat16, weighting_scheme: str = "none",
    guidance_scale: float = 2.5, max_grad_norm: float = 1.0,
):
    from tqdm import tqdm

    global_step = 0
    for epoch in range(num_epochs):
        transformer.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)

        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                loss = flux2_training_step(
                    transformer=transformer, vae=vae, noise_scheduler=noise_scheduler,
                    optimizer=optimizer, lr_scheduler=lr_scheduler,
                    pixel_values=batch["pixel_values"], prompt_embeds=batch["prompt_embeds"],
                    text_ids=batch["text_ids"], accelerator=accelerator, weight_dtype=weight_dtype,
                    weighting_scheme=weighting_scheme, guidance_scale=guidance_scale, max_grad_norm=max_grad_norm,
                )

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            progress_bar.set_postfix(loss=loss)

        progress_bar.close()

    return global_step
