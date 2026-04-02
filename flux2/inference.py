"""
Minimal Flux2 (FLUX.2) inference — the complete sampling algorithm.
Uses the minimal transformer (flux2/model.py) and VAE (flux2/vae.py) from this repo.

Key differences from FLUX.1 (flux1/inference.py):
- VAE decode: Flux2AutoEncoder handles BatchNorm inv_normalize + unpatchify internally
- Timestep shift: compute_empirical_mu (fitted linear, not calculate_shift)
- Position IDs: 4D (T, H, W, L) not 3D (ch, H, W)
- Transformer: no pooled_projections, guidance always on

References (source of truth):
1) BFL flux2-inference — generalized_time_snr_shift, get_schedule, compute_empirical_mu, denoise:
   https://github.com/black-forest-labs/flux2/blob/main/src/flux2/sampling.py
2) BFL Flux2 autoencoder — decode (inv_normalize + unpatchify):
   https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py
"""

import numpy as np
import torch

from utils.training import euler_step
from flux2.training import pack_latents, unpack_latents, prepare_latent_ids


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def get_sigmas_flux2(num_inference_steps, image_seq_len, num_steps):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    shift = np.exp(compute_empirical_mu(image_seq_len, num_steps))
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    sigmas = np.append(sigmas, 0.0)
    return torch.from_numpy(sigmas).float()


@torch.no_grad()
def flux2_inference(
    transformer, vae, prompt_embeds: torch.Tensor, text_ids: torch.Tensor,
    height: int = 1024, width: int = 1024, num_inference_steps: int = 50,
    guidance_scale: float = 2.5, device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16, generator: torch.Generator = None,
):
    device = device or prompt_embeds.device
    latent_height = 2 * (height // (vae.vae_scale_factor * 2))
    latent_width = 2 * (width // (vae.vae_scale_factor * 2))
    num_channels = transformer.in_channels // 4

    latents = torch.randn(1, num_channels * 4, latent_height // 2, latent_width // 2, device=device, dtype=dtype, generator=generator)
    latent_ids = prepare_latent_ids(latents).to(device=device, dtype=dtype)
    latents = pack_latents(latents)

    image_seq_len = latents.shape[1]
    sigmas = get_sigmas_flux2(num_inference_steps, image_seq_len, num_inference_steps).to(device)
    timesteps = sigmas[:-1] * 1000

    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)

    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(dtype)
        noise_pred = transformer(
            hidden_states=latents, timestep=timestep / 1000, guidance=guidance,
            encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_ids,
        )
        latents = euler_step(noise_pred, sigmas[i], sigmas[i + 1], latents)

    return vae.decode(unpack_latents(latents, latent_height // 2, latent_width // 2))
