# minFLUX

![minFLUX](assets/flux.jpg)
An unofficial minimal implementation of [FLUX](https://bfl.ai/models/flux-2) diffusion transformers, like [minGPT](https://github.com/karpathy/minGPT) but for FLUX. For learning only (cannot be used with pretrained weights). Since there are numerous possible design decisions in diffusion models, the purpose of this project is to understand the key model choices in FLUX.

## Structure

```
minFLUX/
├── shared/                         # Dependency-free utilities (only torch + numpy)
│   ├── training_utils.py           # Timestep sampling, loss weighting, get_sigmas
│   ├── latent_utils.py             # Pack/unpack latents, prepare position IDs
│   └── rotary_emb.py               # RoPE: get_1d_rotary_pos_embed, apply_rotary_emb
├── flux1/                          # FLUX.1 (dev/schnell)
│   ├── model.py                    # FluxTransformer2DModel — full architecture (~300 lines)
│   ├── training.py                 # Training: VAE encode -> flow matching -> velocity MSE
│   ├── inference.py                # Inference: noise -> Euler ODE denoise -> VAE decode
│   └── kontext_training.py         # Kontext: reference-image conditioned training
└── flux2/                          # FLUX.2
    ├── model.py                    # Flux2Transformer2DModel — SwiGLU, shared modulation (~350 lines)
    ├── training.py                 # Training: patchify -> BatchNorm -> flow matching
    └── inference.py                # Inference: denoise -> BN denorm -> unpatchify -> decode
```

Each `.py` file has a companion `.md` file, containing documentation and source-of-truth line mappings to the [diffusers](https://github.com/huggingface/diffusers/tree/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a) repo. These are to understand the implementation and verify the code.

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

No external dependencies beyond PyTorch. All utilities (RoPE, latent packing, timestep embeddings) are in `shared/`.

## Warning

This implementation of FLUX.1 and FLUX.2 is inferred from the diffusers repo and may contain errors. Possible sources of inaccuracy include:

- **AI-assisted**: The code is written with the help of AI, referencing the diffusers repo. It was verified line-by-line against the source but not executed end-to-end. This means that the code may contain errors.
- **Diffusers code changes**: Source-of-truth line numbers reference a specific [commit](https://github.com/huggingface/diffusers/tree/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a). The diffusers codebase changes frequently, so functions may move, rename, or change signature.
- **Simplifications**: Stripping ControlNet, IP-Adapter, gradient checkpointing, KV caching, FSDP/DeepSpeed support, and the attention processor dispatch pattern may introduce subtle incompatibilities with pretrained weights. Hence this will not work with pretrained weights. Also the minimal model classes (`flux1/model.py`, `flux2/model.py`) use different attribute names than diffusers' `FluxTransformer2DModel` / `Flux2Transformer2DModel`, so `state_dict` keys will not match directly.
- **FLUX.2 is new**: The FLUX.2 architecture was added to diffusers recently and may still be evolving. The Flux2 files here reflect a snapshot of the codebase at the time of writing.

For verification, cross-reference with the [diffusers source](https://github.com/huggingface/diffusers/tree/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a) and the companion `.md` files for the line mappings.
