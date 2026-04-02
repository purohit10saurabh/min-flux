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
from einops import rearrange

from utils.training import sample_flow_match_noise, flow_match_loss_step, train_loop


def prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def pack_latents(latents):
    return rearrange(latents, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)


def unpack_latents(latents, height, width, vae_scale_factor):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    return rearrange(latents, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                     h=height // 2, w=width // 2, p1=2, p2=2)


def flux_training_step(
    transformer, vae, optimizer, lr_scheduler,
    pixel_values: torch.Tensor, prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor, text_ids: torch.Tensor,
    accelerator, weight_dtype: torch.dtype,
    weighting_scheme: str = "none", guidance_scale: float = 3.5, max_grad_norm: float = 1.0,
):
    model_input = vae.encode(pixel_values).to(dtype=weight_dtype)
    latent_image_ids = prepare_latent_image_ids(
        model_input.shape[2] // 2, model_input.shape[3] // 2, accelerator.device, weight_dtype,
    )
    noisy_model_input, noise, sigmas, timesteps = sample_flow_match_noise(model_input, weighting_scheme)
    bsz = model_input.shape[0]
    packed_noisy_model_input = pack_latents(noisy_model_input)

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

    return flow_match_loss_step(model_pred, noise, model_input, sigmas, weighting_scheme,
                                accelerator, transformer, optimizer, lr_scheduler, max_grad_norm)


def flux_training(
    transformer, vae, optimizer, lr_scheduler,
    train_dataloader, accelerator, num_epochs: int,
    weight_dtype: torch.dtype = torch.bfloat16, weighting_scheme: str = "none",
    guidance_scale: float = 3.5, max_grad_norm: float = 1.0,
):
    step = lambda batch: flux_training_step(
        transformer=transformer, vae=vae, optimizer=optimizer, lr_scheduler=lr_scheduler,
        pixel_values=batch["pixel_values"], prompt_embeds=batch["prompt_embeds"],
        pooled_prompt_embeds=batch["pooled_prompt_embeds"], text_ids=batch["text_ids"],
        accelerator=accelerator, weight_dtype=weight_dtype,
        weighting_scheme=weighting_scheme, guidance_scale=guidance_scale, max_grad_norm=max_grad_norm,
    )
    return train_loop(step, transformer, train_dataloader, accelerator, num_epochs)
