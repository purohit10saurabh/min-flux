"""
Minimal Flux (FLUX.1) inference loop — the complete sampling algorithm.

References (source of truth):
1) BFL flux-inference — time_shift, get_schedule, denoise:
   https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py
2) diffusers FluxPipeline — calculate_shift, timestep prep, VAE decode:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
3) diffusers FluxTransformer2DModel — forward() signature and timestep*1000 convention:
   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
"""

import numpy as np
import torch

from shared.latent_utils import prepare_latent_image_ids, pack_latents, unpack_latents


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + (base_shift - m * base_seq_len)


def get_sigmas(num_inference_steps, image_seq_len):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    shift = np.exp(calculate_shift(image_seq_len))
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    sigmas = np.append(sigmas, 0.0)
    return torch.from_numpy(sigmas).float()


def euler_step(model_output, sigma, sigma_next, sample):
    return sample + (sigma_next - sigma) * model_output


@torch.no_grad()
def flux_inference(
    transformer, vae, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor, height: int = 1024, width: int = 1024,
    num_inference_steps: int = 28, guidance_scale: float = 3.5,
    device: torch.device = None, dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator = None,
):
    device = device or prompt_embeds.device
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))
    num_channels = transformer.config.in_channels

    latents = torch.randn(1, num_channels, latent_height, latent_width, device=device, dtype=dtype, generator=generator)
    latent_image_ids = prepare_latent_image_ids(1, latent_height // 2, latent_width // 2, device, dtype)
    latents = pack_latents(latents, 1, num_channels, latent_height, latent_width)

    image_seq_len = latents.shape[1]
    sigmas = get_sigmas(num_inference_steps, image_seq_len).to(device)
    timesteps = sigmas[:-1] * 1000

    guidance = None
    if transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)

    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(dtype)
        noise_pred = transformer(
            hidden_states=latents, timestep=timestep / 1000, guidance=guidance,
            pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, img_ids=latent_image_ids, return_dict=False,
        )[0]
        latents = euler_step(noise_pred, sigmas[i], sigmas[i + 1], latents)

    latents = unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    return vae.decode(latents.to(vae.dtype), return_dict=False)[0]
