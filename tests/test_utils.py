import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from flux1.training import pack_latents, unpack_latents, prepare_latent_image_ids
from utils.model import (
    get_timestep_embedding, TimestepEmbedding, get_1d_rotary_pos_embed, apply_rotary_emb,
    PosEmbed, AdaLayerNormContinuous, joint_attention,
)
from utils.training import sample_flow_match_noise, euler_step
from flux1.inference import get_sigmas as get_inference_sigmas
from flux2.training import pack_latents as pack_latents_v2, unpack_latents as unpack_latents_v2


@pytest.mark.parametrize("B,C,H,W", [(1, 16, 8, 8), (2, 4, 16, 12), (1, 32, 4, 4)])
def test_flux1_pack_unpack_roundtrip(B, C, H, W):
    vae_scale_factor = 8
    latents = torch.randn(B, C, H, W)
    packed = pack_latents(latents)
    assert packed.shape == (B, (H // 2) * (W // 2), C * 4)
    unpacked = unpack_latents(packed, H * vae_scale_factor, W * vae_scale_factor, vae_scale_factor)
    assert torch.allclose(latents, unpacked)


@pytest.mark.parametrize("B,C,H,W", [(1, 128, 4, 4), (2, 32, 8, 6)])
def test_flux2_pack_unpack_roundtrip(B, C, H, W):
    latents = torch.randn(B, C, H, W)
    packed = pack_latents_v2(latents)
    assert packed.shape == (B, H * W, C)
    unpacked = unpack_latents_v2(packed, H, W)
    assert torch.allclose(latents, unpacked)


def test_euler_step_arithmetic():
    sample = torch.tensor([1.0, 2.0, 3.0])
    pred = torch.tensor([0.5, -0.5, 1.0])
    result = euler_step(pred, 0.8, 0.6, sample)
    assert torch.allclose(result, sample + (0.6 - 0.8) * pred)


def test_euler_step_at_equal_sigmas():
    sample = torch.randn(2, 4)
    assert torch.allclose(euler_step(torch.randn(2, 4), 0.5, 0.5, sample), sample)


@pytest.mark.parametrize("steps,seq_len", [(10, 256), (28, 1024), (50, 4096)])
def test_inference_sigmas_shape_and_bounds(steps, seq_len):
    sigmas = get_inference_sigmas(steps, seq_len)
    assert sigmas.shape == (steps + 1,)
    assert sigmas[-1].item() == 0.0
    assert sigmas[0].item() > 0.0


def test_inference_sigmas_monotonic():
    diffs = get_inference_sigmas(28, 256)
    assert (diffs[1:] - diffs[:-1] <= 0).all()


def test_sample_flow_match_noise_shapes():
    model_input = torch.randn(2, 16, 8, 8)
    noisy, noise, sigmas, timesteps = sample_flow_match_noise(model_input)
    assert noisy.shape == model_input.shape
    assert noise.shape == model_input.shape
    assert sigmas.shape == (2, 1, 1, 1)
    assert timesteps.shape == (2,)


def test_sample_flow_match_noise_interpolation():
    model_input = torch.randn(4, 8, 4, 4)
    noisy, noise, sigmas, _ = sample_flow_match_noise(model_input)
    assert torch.allclose(noisy, (1.0 - sigmas) * model_input + sigmas * noise)


@pytest.mark.parametrize("B,dim", [(1, 128), (4, 256), (2, 64)])
def test_timestep_embedding_shape(B, dim):
    assert get_timestep_embedding(torch.rand(B), dim).shape == (B, dim)


def test_timestep_embedding_distinct():
    emb = get_timestep_embedding(torch.tensor([0.1, 0.5, 0.9]), 128)
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.allclose(emb[1], emb[2])


def test_timestep_embedding_module_forward():
    assert TimestepEmbedding(128, 256).forward(torch.randn(2, 128)).shape == (2, 256)


def test_rotary_embed_shape():
    cos, sin = get_1d_rotary_pos_embed(64, 10)
    assert cos.shape == sin.shape == (10, 64)


def test_apply_rotary_identity_at_zero_positions():
    dim, seq_len = 64, 8
    cos, sin = get_1d_rotary_pos_embed(dim, torch.zeros(seq_len))
    x = torch.randn(1, seq_len, 2, dim)
    assert torch.allclose(apply_rotary_emb(x, (cos, sin)), x, atol=1e-5)


def test_apply_rotary_preserves_shape():
    dim, seq_len = 128, 16
    cos, sin = get_1d_rotary_pos_embed(dim, seq_len)
    x = torch.randn(2, seq_len, 4, dim)
    assert apply_rotary_emb(x, (cos, sin)).shape == x.shape


def test_latent_image_ids_shape_and_values():
    ids = prepare_latent_image_ids(4, 6, "cpu", torch.float32)
    assert ids.shape == (24, 3)
    assert (ids[:, 0] == 0).all()
    assert ids[:, 1].max().item() == 3.0
    assert ids[:, 2].max().item() == 5.0


@pytest.mark.parametrize("theta,axes_dim,seq_len", [
    (10000, (16, 56, 56), 12),
    (2000, (32, 32, 32, 32), 8),
])
def test_pos_embed_shape(theta, axes_dim, seq_len):
    pe = PosEmbed(theta=theta, axes_dim=axes_dim)
    ids = torch.zeros(seq_len, len(axes_dim))
    cos, sin = pe(ids)
    assert cos.shape == sin.shape == (seq_len, sum(axes_dim))


def test_ada_layer_norm_continuous_shape():
    norm = AdaLayerNormContinuous(dim=64, conditioning_dim=64)
    x = torch.randn(2, 10, 64)
    cond = torch.randn(2, 64)
    assert norm(x, cond).shape == (2, 10, 64)


def test_joint_attention_with_context():
    B, seq_img, seq_txt, heads, dim = 1, 8, 4, 2, 32
    q = torch.randn(B, seq_img, heads, dim)
    k = torch.randn(B, seq_img, heads, dim)
    v = torch.randn(B, seq_img, heads, dim)
    q_ctx = torch.randn(B, seq_txt, heads, dim)
    k_ctx = torch.randn(B, seq_txt, heads, dim)
    v_ctx = torch.randn(B, seq_txt, heads, dim)
    out = joint_attention(q, k, v, q_ctx, k_ctx, v_ctx)
    assert out.shape == (B, seq_txt + seq_img, heads, dim)


def test_joint_attention_without_context():
    B, seq, heads, dim = 2, 6, 4, 16
    q = torch.randn(B, seq, heads, dim)
    k = torch.randn(B, seq, heads, dim)
    v = torch.randn(B, seq, heads, dim)
    out = joint_attention(q, k, v, None, None, None)
    assert out.shape == (B, seq, heads, dim)
