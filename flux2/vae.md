# Flux2 VAE Documentation

## Comparison with FLUX.1


| Aspect                           | FLUX.1                            | FLUX.2                       |
| -------------------------------- | --------------------------------- | ---------------------------- |
| z_channels                       | 16                                | 32                           |
| Latent normalization             | `scale_factor` / `shift_factor`   | Patchify (2x2) + `BatchNorm` |
| Sampling                         | Diagonal Gaussian (mean + logvar) | Mean-only (no sampling)      |
| `quant_conv` / `post_quant_conv` | No                                | Yes (`1x1` `Conv2d`)         |


## Overview

This document explains `flux2/vae.py`: a minimal FLUX.2 autoencoder. `Flux2AutoEncoder` wraps the same shared `Encoder` and `Decoder` as FLUX.1 but adds `quant_conv` / `post_quant_conv` on the wrapper, mean-only latents (no logvar sampling), and patchify (2x2) plus `BatchNorm2d` normalization instead of scalar scale/shift.

`Flux2AutoEncoder` (L26-72): wrapper around shared `Encoder`/`Decoder`.

- Constructor (L27-51): `Encoder` plus `quant_conv(2*z_ch, 2*z_ch, 1)`, `Decoder` plus `post_quant_conv(z_ch, z_ch, 1)`, `BatchNorm2d(z_ch*4, affine=False)`. `ps=(2,2)` for patchify.
- `_patchify(z)` (L53-54): maps `(B, C, H, W)` to `(B, C*4, H/2, W/2)` via `einops.rearrange`, matching BFL exactly.
- `_unpatchify(z)` (L56-57): inverse of `_patchify` via `einops.rearrange`.
- `encode(x)` (L59-64): `encoder` → `quant_conv` → take mean half of channels → `_patchify` → `BatchNorm` forward (with `bn.eval()`).
- `decode(z)` (L66-72): `inv_normalize` via running mean/variance (`z * std + mean`) → `_unpatchify` → `post_quant_conv` → `decoder`.

---

## Source of Truth

### Canonical Source Files


| Short Name    | Full Path                                  |
| ------------- | ------------------------------------------ |
| `autoencoder` | [`src/flux2/autoencoder.py`](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/autoencoder.py) |
| `util`        | [`src/flux2/util.py`](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/util.py)        |


### Line-by-Line Mapping


| minFLUX function / block                                       | Canonical Source                | Source Lines     | Verdict                                                                                                                            |
| -------------------------------------------------------------- | ------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `Flux2AutoEncoder.__init_`_ (`Encoder`, `Decoder`, `bn`, `ps`) | `AutoEncoder.__init__`          | 271-302          | MATCH (`quant_conv` / `post_quant_conv` live on minFLUX wrapper; BFL attaches them inside `Encoder` / `Decoder`, 108-181, 184-268) |
| `_patchify` / `_unpatchify`                                    | `encode` / `decode` `rearrange` | 318-323, 329-334 | EXACT MATCH (`rearrange` patterns identical to BFL)                                                                                 |
| `encode` (`quant_conv`, mean chunk, patchify, `bn`)            | `AutoEncoder.encode`            | 314-325          | EXACT MATCH (composition; conv placement differs as noted)                                                                         |
| `decode` `inv_normalize`                                       | `AutoEncoder.inv_normalize`     | 308-312          | EXACT MATCH (same `eps` on variance sqrt)                                                                                          |
| `decode` (`unpatchify`, `post_quant_conv`, `decoder`)          | `AutoEncoder.decode`            | 327-336          | EXACT MATCH                                                                                                                        |
| Default `z_channels=32`                                        | `AutoEncoderParams`             | 10-17            | MATCH (`util` `AutoEncoder(AutoEncoderParams())` uses these defaults)                                                              |


### Notes

- **Shared backbone**: Moving `quant_conv` and `post_quant_conv` out of `Encoder`/`Decoder` lets minFLUX reuse `utils.vae.Encoder` and `utils.vae.Decoder` for both FLUX.1 and FLUX.2; BFL keeps those convs inside the respective module classes.
- **Patchify**: Both BFL and minFLUX use `einops.rearrange` with identical patterns for patchify/unpatchify.
- `**bn.eval()` in `encode`/`decode`**: Matches BFL `normalize` / `inv_normalize` always running the batch norm in eval mode for fixed running statistics.

