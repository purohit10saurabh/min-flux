# Flux1 VAE Documentation

## Overview

This document explains `flux1/vae.py`: a minimal FLUX.1 autoencoder. `FluxAutoEncoder` is a thin wrapper around the shared `Encoder` and `Decoder` in `utils/vae_utils.py`. It applies the BFL latent affine normalization (`scale_factor`, `shift_factor`), optional diagonal-Gaussian sampling on the encoder output, and delegates pixel reconstruction to the decoder.

`FluxAutoEncoder` (L18-49): thin wrapper around shared `Encoder`/`Decoder` from `utils/vae_utils.py`.

- Constructor (L19-38): builds `Encoder`, `Decoder`. Stores `scale_factor=0.3611`, `shift_factor=0.1159`, `vae_scale_factor=8`. Default `z_channels=16`.
- `encode(x, sample=True)` (L40-46): encoder output is split into mean and logvar; if `sample` is true, draws from the diagonal Gaussian, else uses the mean; then applies `scale_factor * (z - shift_factor)`. Returns the scaled latent directly. The `sample` flag supports stochastic latents for training targets and deterministic encoding for Kontext-style reference latents.
- `decode(z)` (L48-49): applies inverse affine `z / scale_factor + shift_factor`, then the decoder. Returns a pixel tensor directly.

---

## Source of Truth

### Canonical Source Files


| Short Name    | Full Path                                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `autoencoder` | `[src/flux/modules/autoencoder.py](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/modules/autoencoder.py)` |
| `util`        | `[src/flux/util.py](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/util.py)`                               |


### Line-by-Line Mapping


| minFLUX function / block                                              | Canonical Source                         | Source Lines                                                             | Verdict                                                                                   |
| --------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `FluxAutoEncoder.__init__` (`Encoder`, `Decoder`, `vae_scale_factor`) | `AutoEncoder.__init__`                   | 282-306                                                                  | MATCH (shared `utils.vae` backbone; BFL defines `Encoder`/`Decoder` in same file 109-264) |
| `scale_factor`, `shift_factor`, `z_channels`                          | `AutoEncoder` fields; `util` `ae_params` | `autoencoder` 305-306; `util` 318-327 (representative `ae_params` block) | MATCH                                                                                     |
| `encode` mean/logvar split + stochastic branch                        | `DiagonalGaussian.forward`               | 267-279                                                                  | EXACT MATCH (inlined; BFL uses fixed `sample_z` on module)                                |
| `encode` affine normalize                                             | `AutoEncoder.encode`                     | 308-311                                                                  | EXACT MATCH                                                                               |
| `decode` inverse affine + `Decoder`                                   | `AutoEncoder.decode`                     | 313-315                                                                  | EXACT MATCH                                                                               |
| `if __name__ == "__main__"` param count demo                          | —                                        | —                                                                        | LOCAL                                                                                     |


### Notes

- **DiagonalGaussian**: BFL wraps sampling in `DiagonalGaussian` with a constructor `sample` flag. Here the same logic lives inside `encode(..., sample=True)` so one module can do training (stochastic) and deterministic encoding without swapping submodules.
- **Backbone**: `Encoder` and `Decoder` implementations are shared with other minFLUX code via `utils/vae_utils.py`; behavior is intended to align with BFL `Encoder`/`Decoder` in `autoencoder.py`.

