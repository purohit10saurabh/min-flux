"""
Minimal Flux2 (FLUX.2) autoencoder — the complete VAE architecture.
Key differences from FLUX.1 (flux1/vae.py):
- quant_conv (1x1) after encoder, post_quant_conv (1x1) before decoder
- Mean-only latent (no DiagonalGaussian sampling)
- Patchify (2x2) + BatchNorm normalization (not scale/shift)
- z_channels=32 (not 16)

References (source of truth):
1) BFL Flux2 autoencoder — AutoEncoder, patchify, BatchNorm normalize/inv_normalize:
   https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py
2) BFL Flux2 model configs — AutoEncoderParams defaults (z_channels=32):
   https://github.com/black-forest-labs/flux2/blob/main/src/flux2/util.py
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from utils.vae_utils import Encoder, Decoder


class Flux2AutoEncoder(nn.Module):
    def __init__(
        self,
        resolution=256,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        z_channels=32,
    ):
        super().__init__()
        ch_mult = list(ch_mult)
        self.z_channels = z_channels
        self.ps = (2, 2)
        self.vae_scale_factor = 2 ** (len(ch_mult) - 1)

        self.encoder = Encoder(resolution, in_channels, ch, ch_mult, num_res_blocks, z_channels)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.decoder = Decoder(ch, out_ch, ch_mult, num_res_blocks, in_channels, resolution, z_channels)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * z_channels,
            eps=1e-4, momentum=0.1, affine=False, track_running_stats=True,
        )

    def _patchify(self, z: Tensor) -> Tensor:
        return rearrange(z, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=self.ps[0], pj=self.ps[1])

    def _unpatchify(self, z: Tensor) -> Tensor:
        return rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=self.ps[0], pj=self.ps[1])

    def encode(self, x: Tensor) -> Tensor:
        moments = self.quant_conv(self.encoder(x))
        mean = moments.chunk(2, dim=1)[0]
        z = self._patchify(mean)
        self.bn.eval()
        return self.bn(z)

    def decode(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + 1e-4)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        z = z * s.to(z.device, z.dtype) + m.to(z.device, z.dtype)
        z = self._unpatchify(z)
        return self.decoder(self.post_quant_conv(z))


if __name__ == "__main__":
    model = Flux2AutoEncoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Flux2 AutoEncoder: {n_params / 1e6:.1f}M params")
    print(f"  z_channels=32, patchify={model.ps}, BatchNorm normalization")
    print(f"  vae_scale_factor={model.vae_scale_factor}")
