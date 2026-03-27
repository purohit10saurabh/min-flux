# Flux Training Documentation

## Overview

This document explains the Flux diffusion model training implementation, the theoretical foundations behind each design choice, and references to the original research papers.

## Key Papers

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | Lipman et al. | 2023 (ICLR) | Foundation of flow matching training |
| [Scaling Rectified Flow Transformers (SD3)](https://arxiv.org/abs/2403.03206) | Esser et al. | 2024 | Weighting schemes, architecture |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | Liu et al. | 2022 | Linear interpolation formulation |

---

## Core Concepts

### 1. Rectified Flow vs Traditional Diffusion

**Traditional Diffusion** uses a stochastic forward process that gradually adds noise over many steps following a variance schedule. The model learns to predict the noise added at each step.

**Rectified Flow** connects data and noise distributions via a straight line (linear interpolation), making the trajectory simpler and enabling fewer sampling steps.

```
Traditional Diffusion:  x_t = √(α_t) * x_0 + √(1-α_t) * ε  (curved path)
Rectified Flow:         x_t = (1-t) * x_0 + t * ε          (straight path)
```

**Why Flux uses Rectified Flow:**
- Conceptually simpler
- Better performance with fewer sampling steps
- More stable training
- State-of-the-art results on text-to-image benchmarks

---

## Training Components

### 2. Noisy Input Construction

```python
noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
```

**Formula:** `z_t = (1 - σ) * x + σ * ε`

Where:
- `x` = clean latent (encoded image)
- `ε` = random Gaussian noise
- `σ` = timestep-dependent sigma value ∈ [0, 1]
- `z_t` = noisy latent at timestep t

**Why this formulation?**

From the Rectified Flow paper (Liu et al., 2022), linear interpolation between data and noise creates straight trajectories in the probability path. This is simpler than the curved paths in DDPM-style diffusion and allows the model to learn a direct mapping.

At `σ=0`: `z_t = x` (pure data)
At `σ=1`: `z_t = ε` (pure noise)

---

### 3. The Target Choice: `noise - model_input`

```python
target = noise - model_input
```

**This is the velocity target for flow matching.**

**Mathematical Derivation:**

In rectified flow, we parameterize the flow as:
```
x_t = (1-t) * x_0 + t * x_1
```

Where `x_0` is the data sample and `x_1` is the noise sample.

The velocity (derivative with respect to t) is:
```
v = dx_t/dt = x_1 - x_0 = noise - data
```

**Why predict velocity instead of noise?**

| Approach | Target | Used By |
|----------|--------|---------|
| ε-prediction | `ε` (noise) | DDPM, SD 1.x/2.x |
| v-prediction | `ε - x` (velocity) | SD3, Flux, Imagen Video |
| x-prediction | `x` (clean data) | Some variants |

**Advantages of velocity prediction:**
1. **Balanced gradients**: At intermediate timesteps, both noise and signal contribute equally
2. **Better for few-step sampling**: The straight-line trajectory is easier to approximate
3. **Numerical stability**: Avoids issues at boundary timesteps (t=0 or t=1)

From the SD3 paper: "The rectified flow formulation connects data and noise distributions on a straight line, which is simpler conceptually and performs better with fewer sampling steps."

---

### 4. Timestep Sampling Strategies

```python
def compute_density_for_timestep_sampling(weighting_scheme, batch_size, ...):
```

Not all timesteps are equally important for perceptual quality. The SD3 paper introduces biased timestep sampling to focus training on "perceptually relevant scales."

#### Uniform Sampling (Default)
```python
u = torch.rand(size=(batch_size,))
```
Simple uniform sampling over [0, 1]. Baseline approach.

#### Logit-Normal Sampling
```python
u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,))
u = torch.sigmoid(u)
```

**Why Logit-Normal?**

The logit-normal distribution concentrates samples around the mean (default: 0.0 after sigmoid → 0.5), focusing training on mid-range noise levels where perceptual details are learned.

- Low noise (σ ≈ 0): Image is nearly clean, little to learn
- High noise (σ ≈ 1): Image is nearly pure noise, details invisible
- **Mid-range (σ ≈ 0.5): Critical for perceptual quality**

Parameters:
- `logit_mean=0.0`: Centers sampling at t=0.5
- `logit_std=1.0`: Controls spread

#### Mode Sampling
```python
u = torch.rand(size=(batch_size,))
u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
```

Uses a cosine-based transformation to create a mode (peak) in the sampling distribution. The `mode_scale` parameter (default: 1.29) controls the sharpness of the peak.

---

### 5. Loss Weighting Schemes

```python
def compute_loss_weighting_for_sd3(weighting_scheme, sigmas):
```

Different timesteps may warrant different loss weights to balance training signal.

#### Uniform Weighting (Default)
```python
weighting = torch.ones_like(sigmas)
```
All timesteps weighted equally.

#### Sigma-Sqrt Weighting
```python
weighting = (sigmas ** -2.0).float()
```

Upweights low-noise timesteps. From the theoretical analysis in score-based diffusion, this relates to the score function scaling.

**Intuition:** At low noise, the signal-to-noise ratio is high, and small errors are more perceptually significant.

#### Cosmap Weighting
```python
bot = 1 - 2 * sigmas + 2 * sigmas**2
weighting = 2 / (math.pi * bot)
```

Derived from the cosine noise schedule commonly used in improved DDPM. Provides smooth weighting that emphasizes mid-range timesteps.

**From SD3 paper Section 3.1:** "We improve existing noise sampling techniques for training rectified flow models by biasing them towards perceptually relevant scales."

---

### 6. Flow Matching Loss

```python
loss = torch.mean(
    (weighting * (model_pred - target) ** 2).reshape(target.shape[0], -1),
    1
)
loss = loss.mean()
```

**Simple MSE loss** between predicted and target velocity, optionally weighted.

**Why MSE?**

Flow matching (Lipman et al., 2023) shows that regressing onto the conditional vector field with MSE loss is equivalent to minimizing the KL divergence to the target distribution, without requiring simulation of ODEs during training.

---

## Architecture Notes

### Flux Transformer Components

| Component | Purpose |
|-----------|---------|
| `hidden_states` | Packed latent representation |
| `timestep` | Normalized timestep (divided by 1000) |
| `guidance` | CFG-like guidance embedding |
| `pooled_projections` | Pooled text embeddings |
| `encoder_hidden_states` | Sequence text embeddings |
| `txt_ids` | Text position IDs |
| `img_ids` | Image position IDs |

### Latent Packing/Unpacking

Flux uses a patchified representation where latents are "packed" into sequences for transformer processing:

```python
packed = FluxPipeline._pack_latents(latents, ...)
# After transformer
unpacked = FluxPipeline._unpack_latents(output, ...)
```

This is similar to ViT-style patch embedding but applied in latent space.

---

## Training Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| `weighting_scheme` | `"logit_normal"` | From SD3 paper |
| `logit_mean` | 0.0 | Centers on t=0.5 |
| `logit_std` | 1.0 | Standard spread |
| `mode_scale` | 1.29 | For mode sampling |
| `guidance_scale` | 3.5 | Training guidance |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `learning_rate` | 1e-4 to 1e-6 | Depends on fine-tuning vs training |

---

## Summary

The Flux training implements **rectified flow matching** with:

1. **Linear interpolation** between data and noise
2. **Velocity prediction** target: `v = ε - x`
3. **Biased timestep sampling** focusing on perceptually relevant scales
4. **Optional loss weighting** to balance training signal
5. **Simple MSE loss** without simulation

These choices follow state-of-the-art practices from the SD3 paper and enable high-quality text-to-image generation with efficient training and sampling.

---

## Source of Truth

Every function in `flux1/training.py` maps to a canonical diffusers source. Shared utilities live in `utils/training.py`. Verified against the `diffusers` repo source code.

### Why DreamBooth as Source of Truth?

Diffusers has no standalone "train Flux" script. All 12 official Flux training examples live under `examples/dreambooth/` or `examples/controlnet/`. The core library (`src/diffusers/`) provides the building blocks -- `compute_density_for_timestep_sampling` in `training_utils.py`, `_pack_latents` in `pipeline_flux.py`, `forward()` in `transformer_flux.py` -- but never assembles them into a training step. The velocity target `noise - model_input` does not appear anywhere in `src/diffusers/`, only in the example scripts.

The dreambooth script is the canonical composition: VAE encode, noise interpolation, pack, transformer forward, unpack, velocity MSE. The DreamBooth-specific parts (prior preservation, class images, LoRA setup) are omitted in minFLUX. `train_dreambooth_lora_flux.py` was chosen over the other 11 scripts because it is the most widely referenced.

### Canonical Source Files

| Short Name | Full Path |
|------------|-----------|
| `training` | `diffusers/src/diffusers/training.py` |
| `dreambooth_flux` | `diffusers/examples/dreambooth/train_dreambooth_lora_flux.py` |
| `pipeline_flux` | `diffusers/src/diffusers/pipelines/flux/pipeline_flux.py` |
| `transformer_flux` | `diffusers/src/diffusers/models/transformers/transformer_flux.py` |

### Line-by-Line Mapping

| minFLUX function / block | Lines | Canonical Source | Source Lines | Verdict |
|---------------------------|-------|------------------|--------------|---------|
| `compute_density_for_timestep_sampling` | 19-36 | `training` | 360-384 | EXACT MATCH |
| `compute_loss_weighting_for_sd3` | 39-47 | `training` | 387-402 | EXACT MATCH |
| `get_sigmas` | 50-58 | `dreambooth_flux` (inner fn) | 1678-1687 | MATCH (refactored from closure) |
| VAE encode + shift/scale | 82-84 | `dreambooth_flux` | 1744-1748 | EXACT MATCH |
| `_prepare_latent_image_ids` call | 86-93 | `dreambooth_flux` calling `pipeline_flux` | 1750-1758 / 506-518 | EXACT MATCH |
| Timestep sampling (u -> indices -> timesteps) | 98-106 | `dreambooth_flux` | 1763-1773 | EXACT MATCH |
| Noise interpolation `(1-σ)*x + σ*ε` | 108-109 | `dreambooth_flux` | 1775-1778 | EXACT MATCH |
| `_pack_latents` call | 111-117 | `dreambooth_flux` calling `pipeline_flux` | 1780-1786 / 520-526 | EXACT MATCH |
| Guidance embedding | 119-121 | `dreambooth_flux` | 1788-1793 | MINOR DIFF: no `unwrap_model` |
| Transformer forward | 123-132 | `dreambooth_flux` + `transformer_flux` forward | 1795-1806 / 637-653 | EXACT MATCH |
| `_unpack_latents` call | 134-139 | `dreambooth_flux` calling `pipeline_flux` | 1807-1812 / 528-542 | EXACT MATCH |
| Loss weighting + target + MSE | 141-144 | `dreambooth_flux` | 1814-1840 | EXACT MATCH (minus prior_preservation) |
| Optimizer step | 146-151 | Standard Accelerate pattern | N/A | CORRECT |

### Notes

- **`timestep / 1000`**: The transformer internally multiplies by 1000 (`transformer_flux.py` line 688). The division before calling and multiplication inside cancel out, preserving the original timestep value for the sinusoidal embedding.
- **Guidance**: Diffusers uses `unwrap_model(transformer).config.guidance_embeds` for FSDP/DeepSpeed compatibility. This minimal version accesses `transformer.config` directly (single-GPU only).
- **Prior preservation**: Diffusers dreambooth supports `with_prior_preservation` loss chunking (lines 1821-1844). Omitted here for minimality.

---

## References

1. Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. ICLR 2023. https://arxiv.org/abs/2210.02747

2. Liu, X., Gong, C., & Liu, Q. (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. https://arxiv.org/abs/2209.03003

3. Esser, P., Kulal, S., Blattmann, A., et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. https://arxiv.org/abs/2403.03206

4. Black Forest Labs. (2024). FLUX Model. https://blackforestlabs.ai/
