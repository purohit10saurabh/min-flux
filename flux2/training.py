"""
Minimal Flux2 (FLUX.2) training — the complete training algorithm.
Uses the minimal transformer (flux2/model.py) and VAE (flux2/vae.py) from this repo.

Key differences from FLUX.1 (flux1/training.py):
- VAE: Flux2AutoEncoder with patchify + BatchNorm normalization (not shift_factor/scaling_factor)
- Latents: patchified (2x2) then flattened (not 2x2-patch-rearranged in one step)
- Position IDs: 4D (T, H, W, L) not 3D (ch, H, W)
- Transformer: no pooled_projections, guidance always on, shared modulation
- Text encoder: Mistral3 (not CLIP+T5)

References (source of truth):
1) BFL Flux2 autoencoder — encode (patchify + BatchNorm), decode (inv_normalize + unpatchify):
   https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py
2) BFL Flux2 sampling — denoise, get_schedule, compute_empirical_mu:
   https://github.com/black-forest-labs/flux2/blob/main/src/flux2/sampling.py
3) diffusers training utilities — timestep density sampling, loss weighting:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
4) SD3 paper (Esser et al., 2024) — rectified flow, weighting schemes:
   https://arxiv.org/abs/2403.03206
"""

import torch
from einops import rearrange

from utils.training import sample_flow_match_noise, flow_match_loss_step, train_loop


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    return rearrange(latents, 'b c h w -> b (h w) c')


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(latents, 'b (h w) c -> b c h w', h=height, w=width)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = latents.shape
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1))
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def flux2_training_step(
    transformer, vae, optimizer, lr_scheduler,
    pixel_values: torch.Tensor, prompt_embeds: torch.Tensor, text_ids: torch.Tensor,
    accelerator, weight_dtype: torch.dtype, weighting_scheme: str = "none",
    guidance_scale: float = 2.5, max_grad_norm: float = 1.0,
):
    model_input = vae.encode(pixel_values).to(dtype=weight_dtype)
    latent_ids = prepare_latent_ids(model_input).to(device=accelerator.device, dtype=weight_dtype)
    noisy_model_input, noise, sigmas, timesteps = sample_flow_match_noise(model_input, weighting_scheme)
    bsz = model_input.shape[0]

    packed_noisy_model_input = pack_latents(noisy_model_input)
    guidance = torch.full([1], guidance_scale, device=accelerator.device, dtype=torch.float32).expand(bsz)

    model_pred = transformer(
        hidden_states=packed_noisy_model_input, timestep=timesteps / 1000, guidance=guidance,
        encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_ids,
    )
    model_pred = unpack_latents(model_pred, model_input.shape[2], model_input.shape[3])

    return flow_match_loss_step(model_pred, noise, model_input, sigmas, weighting_scheme,
                                accelerator, transformer, optimizer, lr_scheduler, max_grad_norm)


def flux2_training(
    transformer, vae, optimizer, lr_scheduler,
    train_dataloader, accelerator, num_epochs: int,
    weight_dtype: torch.dtype = torch.bfloat16, weighting_scheme: str = "none",
    guidance_scale: float = 2.5, max_grad_norm: float = 1.0,
):
    step = lambda batch: flux2_training_step(
        transformer=transformer, vae=vae, optimizer=optimizer, lr_scheduler=lr_scheduler,
        pixel_values=batch["pixel_values"], prompt_embeds=batch["prompt_embeds"],
        text_ids=batch["text_ids"], accelerator=accelerator, weight_dtype=weight_dtype,
        weighting_scheme=weighting_scheme, guidance_scale=guidance_scale, max_grad_norm=max_grad_norm,
    )
    return train_loop(step, transformer, train_dataloader, accelerator, num_epochs)
