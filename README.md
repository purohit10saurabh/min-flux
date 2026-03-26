# min-flux

A minimal way to understand FLUX diffusion transformers. Inspired by [minGPT](https://github.com/karpathy/minGPT).

## Structure

```
min-flux/
├── shared/                         # Dependency-free utilities (only torch + numpy)
│   ├── training_utils.py           # Timestep sampling, loss weighting, get_sigmas
│   ├── latent_utils.py             # Pack/unpack latents, prepare position IDs
│   └── rotary_emb.py               # RoPE: get_1d_rotary_pos_embed, apply_rotary_emb
├── flux1/                          # FLUX.1 (dev/schnell)
│   ├── model.py                    # FluxTransformer2DModel — full architecture (~270 lines)
│   ├── training_loop.py            # Training: VAE encode -> flow matching -> velocity MSE
│   ├── inference_loop.py           # Inference: noise -> Euler ODE denoise -> VAE decode
│   └── kontext_training_loop.py    # Kontext: reference-image conditioned training
└── flux2/                          # FLUX.2
    ├── model.py                    # Flux2Transformer2DModel — SwiGLU, shared modulation (~300 lines)
    ├── training_loop.py            # Training: patchify -> BatchNorm -> flow matching
    └── inference_loop.py           # Inference: denoise -> BN denorm -> unpatchify -> decode
```

Each `.py` has a companion `.md` with source-of-truth line mappings to the [diffusers](https://github.com/huggingface/diffusers) repo.

## Key Equations

**Training** (rectified flow matching):
```
noisy = (1 - sigma) * data + sigma * noise       # linear interpolation
target = noise - data                              # velocity
loss = MSE(model(noisy, t), target)               # weighted MSE
```

**Inference** (Euler ODE step):
```
x_{t-1} = x_t + (sigma_next - sigma) * model(x_t, t)
```

## FLUX.1 vs FLUX.2

| | FLUX.1 | FLUX.2 |
|---|--------|--------|
| Text encoder | CLIP + T5 | Mistral3 |
| VAE norm | shift/scale | BatchNorm |
| FFN | GELU | SwiGLU |
| Modulation | Per-block AdaLN | Shared across blocks |
| RoPE | theta=10000, 3 axes | theta=2000, 4 axes |
| Blocks | 19 double + 38 single | 8 double + 48 single |

## No External Dependencies

Zero imports from `diffusers` or any ML framework beyond PyTorch.
All diffusers utilities (RoPE, latent packing, timestep embeddings) are inlined in `shared/`.

## Warning

This code is for **learning purposes** and may contain errors. Possible sources of inaccuracy:

- **AI-generated**: All code was written by AI, referencing the diffusers repo. It was verified line-by-line against the source but not executed end-to-end.
- **Diffusers code changes**: Source-of-truth line numbers reference a specific commit `cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a` of the diffusers repo. The diffusers codebase changes frequently; functions may move, rename, or change signature.
- **Simplification artifacts**: Stripping ControlNet, IP-Adapter, gradient checkpointing, KV caching, FSDP/DeepSpeed support, and the attention processor dispatch pattern may introduce subtle incompatibilities with pretrained weights.
- **Weight loading**: The minimal model classes (`flux1/model.py`, `flux2/model.py`) use different attribute names than diffusers' `FluxTransformer2DModel` / `Flux2Transformer2DModel`, so `state_dict` keys will not match directly.
- **FLUX.2 is new**: The FLUX.2 architecture was added to diffusers recently and may still be evolving. The Flux2 files here reflect a snapshot of the codebase at the time of writing.

Always cross-reference with the [diffusers source](https://github.com/huggingface/diffusers) and the companion `.md` files for the canonical line mappings.
