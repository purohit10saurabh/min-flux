"""
Minimal Flux (FLUX.1) autoencoder — the complete VAE architecture.

References (source of truth):
1) BFL Flux1 autoencoder — AutoEncoder, DiagonalGaussian, scale/shift normalization:
   https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
2) BFL model configs — ae_params (z_channels=16, scale_factor=0.3611, shift_factor=0.1159):
   https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py
"""

import torch
import torch.nn as nn
from torch import Tensor

from utils.vae_utils import Encoder, Decoder


class FluxAutoEncoder(nn.Module):
    def __init__(
        self,
        resolution=256,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    ):
        super().__init__()
        ch_mult = list(ch_mult)
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        self.vae_scale_factor = 2 ** (len(ch_mult) - 1)

        self.encoder = Encoder(resolution, in_channels, ch, ch_mult, num_res_blocks, z_channels)
        self.decoder = Decoder(ch, out_ch, ch_mult, num_res_blocks, in_channels, resolution, z_channels)

    def encode(self, x: Tensor, sample: bool = True) -> Tensor:
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        if sample:
            z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            z = mean
        return self.scale_factor * (z - self.shift_factor)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z / self.scale_factor + self.shift_factor)


if __name__ == "__main__":
    model = FluxAutoEncoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Flux AutoEncoder: {n_params / 1e6:.1f}M params")
    print(f"  z_channels=16, scale_factor={model.scale_factor}, shift_factor={model.shift_factor}")
    print(f"  vae_scale_factor={model.vae_scale_factor}")
