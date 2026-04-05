# Shared VAE Building Blocks Documentation

## Overview

`utils/vae_utils.py` holds convolutional autoencoder primitives shared by FLUX.1 (`flux1/vae.py`) and FLUX.2 (`flux2/vae.py`): `swish`, `AttnBlock`, `ResnetBlock`, `Downsample`, `Upsample`, `Encoder`, and `Decoder`. The spatial tower matches Black Forest Labs reference autoencoders, using `einops.rearrange` for attention tensor layout (matching the BFL source exactly). `Encoder` ends at `2 * z_channels` moments (mean and log-variance channels before any FLUX.2 `quant_conv`). `Decoder` maps latent channels back to pixels; FLUX.2 applies `post_quant_conv` and patch/BatchNorm logic in `flux2/vae.py`, not in this file.

## Source of Truth

### Canonical Source Files


| Short Name | Full Path                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `flux1_ae` | `[src/flux/modules/autoencoder.py](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/modules/autoencoder.py)` |
| `flux2_ae` | `[src/flux2/autoencoder.py](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/autoencoder.py)`              |


### Line-by-Line Mapping


| minFLUX symbol | Lines    | `flux1_ae`                                   | `flux2_ae` | Verdict                                                                                                                         |
| -------------- | -------- | -------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `swish`        | 18-19    | 21-22                                        | 20-21      | EXACT MATCH                                                                                                                     |
| `AttnBlock`    | 22-42    | 25-52 (`AttnBlock`, `attention` + `forward`) | 24-51      | MATCH (attention inlined into `forward`; `rearrange` calls match BFL exactly)                                                   |
| `ResnetBlock`  | 45-62    | 55-82                                        | 54-81      | MATCH (`forward` compressed; same ops; `out_channels is None` guard removed — all callers pass explicit `int`)                  |
| `Downsample`   | 65-71    | 85-95                                        | 84-94      | MATCH (`pad`+`conv` one-liner)                                                                                                  |
| `Upsample`     | 74-80    | 98-106                                       | 97-105     | MATCH (`interpolate`+`conv` one-liner)                                                                                          |
| `Encoder`      | 83-125   | 109-180                                      | 108-181    | MATCH core tower; stripped unused `resolution` param and empty `down.attn` lists + dead branches; `flux2_ae` `quant_conv` omitted (applied in `flux2/vae.py`) |
| `Decoder`      | 128-172  | 183-264                                      | 184-268    | MATCH core tower; stripped unused `resolution`+`in_channels` params and empty `up.attn` lists + dead branches; `flux2_ae` `post_quant_conv` omitted (applied in `flux2/vae.py`) |


### Notes

- **Attention layout**: BFL and minFLUX both use `rearrange(q, "b c h w -> b 1 (h w) c")` for the `(batch, 1, seq, channels)` layout required by `scaled_dot_product_attention`, then `rearrange` back to `NCHW`.
- **Optional level attention**: BFL builds `down.attn` / `up.attn` as empty `ModuleList`s. minFLUX removes them and their dead `len(...) > 0` forward branches since standard FLUX checkpoints never populate them.
- **FLUX.1 vs FLUX.2 wrapping**: `flux1_ae` pairs `Encoder`/`Decoder` with `DiagonalGaussian` and scale/shift on `AutoEncoder`. `flux2_ae` keeps `quant_conv` / `post_quant_conv` inside the encoder/decoder modules; minFLUX keeps a single `Encoder`/`Decoder` and lets `flux2/vae.py` own the 1x1 convs and patch/BatchNorm path.

