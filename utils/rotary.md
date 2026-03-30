# Rotary Positional Embeddings (RoPE) Documentation

## Overview

This document describes `utils/rotary.py`: real-valued RoPE frequency tables and application for Flux-style transformers. Frequencies are built per position along one axis; cosine and sine are `repeat_interleave`d along the feature dimension to match pairs of real/imaginary slots. Application broadcasts `cos`/`sin` over batch and heads, splits the last dimension into adjacent real/imag pairs, applies the rotation, and fuses back.

---

## Source of Truth

### Canonical Source Files

| Short Name | Full Path |
|------------|-----------|
| `embeddings` | [`src/diffusers/models/embeddings.py`](https://github.com/huggingface/diffusers/blob/cbf4d9a3c384ef97d6b0e40c9846dd9e0e41886a/src/diffusers/models/embeddings.py) |

### Line-by-Line Mapping

| minFLUX function / block | Canonical Source | Source Lines | Verdict |
|----------------------------|------------------|--------------|---------|
| `get_1d_rotary_pos_embed` (int `pos` → `torch.arange`) | `get_1d_rotary_pos_embed` | 1157-1158 | MATCH (same `int` handling) |
| `freqs` pre-magnitude (`theta`, `arange(0, dim, 2)`, `/ dim`) | `get_1d_rotary_pos_embed` | 1162-1165 | MATCH with `ntk_factor=1.0`, `linear_factor=1.0` (reference factors omitted; no-op) |
| `torch.outer(pos.float(), freqs)` | `get_1d_rotary_pos_embed` | 1166 | MATCH (`pos` as tensor; explicit `.float()` on positions) |
| `freqs_cos` / `freqs_sin` via `cos`/`sin` + `repeat_interleave(2, dim=1)` | `get_1d_rotary_pos_embed` | 1170-1174 | MATCH (Flux path: `use_real=True`, `repeat_interleave_real=True`; reference may pass `output_size` on `repeat_interleave`) |
| Omitted branches in `get_1d_rotary_pos_embed` | `get_1d_rotary_pos_embed` | 1155, 1159-1160, 1167-1169, 1175-1183 | STRIPPED (`dim % 2` assert; `np.ndarray` positions; NPU float cast; `use_real` without interleave; complex `torch.polar` path) |
| `apply_rotary_emb` broadcast `sequence_dim==2` | `apply_rotary_emb` | 1209-1211 | EXACT MATCH (`cos`/`sin` as `[None, None, :, :]`) |
| `apply_rotary_emb` broadcast `sequence_dim==1` | `apply_rotary_emb` | 1212-1214 | EXACT MATCH (`[None, :, None, :]`) |
| `cos`/`sin` `.to(x.device)` | `apply_rotary_emb` | 1218 | EXACT MATCH |
| `reshape` + `unbind(-1)` + `stack` + `flatten(3)` rotation | `apply_rotary_emb` | 1220-1223 | EXACT MATCH (Flux path: `use_real_unbind_dim=-1`) |
| `(x.float() * cos + x_rotated.float() * sin).to(x.dtype)` | `apply_rotary_emb` | 1231-1233 | EXACT MATCH |
| Omitted branches in `apply_rotary_emb` | `apply_rotary_emb` | 1215-1216, 1224-1229, 1234-1240 | STRIPPED (invalid `sequence_dim` raise; `use_real_unbind_dim==-2` path; `use_real=False` / complex path) |

### Notes

- **Flux-only surface**: The reference `get_1d_rotary_pos_embed` implements Lumina-style complex `freqs_cis`, Stable-Audio-style duplicated cos/sin blocks, and NTK/linear scaling. minFLUX keeps only the branch documented as Flux/Hunyuan-DiT/CogVideoX (`use_real=True`, `repeat_interleave_real=True`). Default keyword arguments on the minFLUX signature mirror the reference defaults but the body always follows that single path.
- **`apply_rotary_emb`**: The reference fixes `use_real=True` and the Flux rotation layout via `use_real_unbind_dim=-1`. minFLUX hard-codes that path and supports `sequence_dim` 1 (joint attention layout) and 2 only; other values follow the `sequence_dim==2` broadcasting branch without raising like diffusers.
