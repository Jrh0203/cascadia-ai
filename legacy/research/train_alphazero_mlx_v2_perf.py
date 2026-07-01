#!/usr/bin/env python3
"""MLX trainer for the Cascadia AlphaZero v2 hybrid network.

v2 architecture:
  • Compact 127-cell hex disk (radius-6, 99.9% empirical coverage).
  • Hex-aware ConvResNet trunk (7-tap conv: self + 6 hex neighbors).
  • Set-Transformer entity stream (2 SAB blocks, 4 heads, 8 tokens × 64 dim).
  • Cross-attention board ← entities (2 heads, residual into trunk).
  • Multi-head value head (16 sub-heads, 3-way phase gate → scalar in [0,1]).
  • Factorized policy over candidate moves (tile-cell, wildlife-cell,
    market-slot, wildlife-market-slot, skip-wildlife).

Reads `AZD2` sample files written by `cascadia-cli --az-collect --az-arch v2`.
Writes `AZR2` weights consumed by `cascadia-cli --az` (auto-detected).

Loss = KL on factorized policy + MSE on scalar value + 0.3 × MSE on 16 aux heads.
"""

from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "MLX is not installed for this Python. Install with:\n"
        "  python3 -m pip install mlx\n"
        "or use the Codex bundled Python shown in docs/ALPHAZERO_MLX.md"
    ) from exc


AZD_MAGIC_V2 = b"AZD2"
AZR_MAGIC_V2 = b"AZR2"

# ── Architectural constants (must match crates/cascadia-ai/src/alphazero_v2.rs) ──

AZ_LOCAL_CELLS = 127
AZ_PAD_INDEX = 127
AZ_CELLS_PADDED = 128
AZ_INPUT_CHANNELS_V2 = 68   # Phase 0.5: was 72; no broadcast scalars.
AZ_TRUNK_CHANNELS_DEFAULT = 96
AZ_TRUNK_BLOCKS_DEFAULT = 6
AZ_ENTITY_TOKENS = 10        # Phase 0.5: was 8; added Globals + RaceState tokens.
AZ_ENTITY_RAW_DIM = 32
AZ_ENTITY_DIM_DEFAULT = 64
AZ_ATTN_HEADS_DEFAULT = 4
AZ_SAB_BLOCKS_DEFAULT = 2
AZ_SAB_FFN_DIM = 128
AZ_CROSS_HEADS = 2
AZ_VALUE_HIDDEN_DEFAULT = 128
AZ_VALUE_SUBHEADS = 16
AZ_VALUE_PHASES = 3
AZ_VALUE_SCALE = 120.0

# Shared opponent trunk (Phase 0.7): each of 3 opponents is encoded with the
# same HexConv stack and its pooled vector replaces entity tokens 4..7 at
# forward time.
AZ_OPP_TRUNK_CHANNELS = 32
AZ_OPP_TRUNK_BLOCKS = 3
AZ_MAX_OPPONENTS = 3
assert AZ_OPP_TRUNK_CHANNELS == AZ_ENTITY_RAW_DIM, \
    "opp trunk channels must equal entity raw dim so pooled output slots directly in"

# Token indices in the Phase 0.5 entity stream.
TOKEN_MARKET_START = 0   # tokens 0..4
TOKEN_OPP_START = 4      # tokens 4..7
TOKEN_BAG = 7
TOKEN_GLOBALS = 8
TOKEN_RACE = 9

# Per-head aux-value scales (must match Rust AZ_AUX_SCALES).
AZ_AUX_SCALES = np.array(
    [13.0, 13.0, 25.0, 25.0, 12.0,
     15.0, 15.0, 15.0, 15.0, 15.0,
     5.0,
     3.0, 3.0, 3.0, 3.0, 3.0],
    dtype=np.float32,
)
assert AZ_AUX_SCALES.shape == (AZ_VALUE_SUBHEADS,)

LN_EPS = 1e-5


# ── Hex disk lookup (radius-6 spiral) — must match nnue.rs::v6_peak ──

GRID_DIM = 21
GRID_CENTER = 10
V6_RADIUS = 6


def _hex_dist_from_center(col: int, row: int) -> int:
    q = col - GRID_CENTER
    r = row - GRID_CENTER
    return max(abs(q), abs(r), abs(q + r))


def _build_local_to_global() -> np.ndarray:
    out = np.zeros(AZ_LOCAL_CELLS, dtype=np.int32)
    nxt = 0
    for d in range(V6_RADIUS + 1):
        for col in range(GRID_DIM):
            for row in range(GRID_DIM):
                if _hex_dist_from_center(col, row) == d:
                    out[nxt] = col * GRID_DIM + row
                    nxt += 1
    assert nxt == AZ_LOCAL_CELLS, f"spiral built {nxt} cells, expected {AZ_LOCAL_CELLS}"
    return out


def _build_global_to_local(l2g: np.ndarray) -> np.ndarray:
    out = -np.ones(GRID_DIM * GRID_DIM, dtype=np.int32)
    for li, gi in enumerate(l2g):
        out[gi] = li
    return out


# Axial-coord neighbors: E, NE, NW, W, SW, SE.
_HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def _build_hex_neighbors() -> np.ndarray:
    """(128, 7) int32 — self + 6 neighbors. Out-of-disk → AZ_PAD_INDEX (127)."""
    l2g = _build_local_to_global()
    g2l = _build_global_to_local(l2g)
    out = np.full((AZ_CELLS_PADDED, 7), AZ_PAD_INDEX, dtype=np.int32)
    for li in range(AZ_LOCAL_CELLS):
        gi = int(l2g[li])
        col = gi // GRID_DIM
        row = gi % GRID_DIM
        q = col - GRID_CENTER
        r = row - GRID_CENTER
        out[li, 0] = li  # self
        for d, (dq, dr) in enumerate(_HEX_DIRS):
            nq = q + dq
            nr = r + dr
            ncol = nq + GRID_CENTER
            nrow = nr + GRID_CENTER
            if 0 <= ncol < GRID_DIM and 0 <= nrow < GRID_DIM:
                ngi = ncol * GRID_DIM + nrow
                nli = int(g2l[ngi])
                if nli >= 0:
                    out[li, 1 + d] = nli
                else:
                    out[li, 1 + d] = AZ_PAD_INDEX
            else:
                out[li, 1 + d] = AZ_PAD_INDEX
    # Pad cell (index 127) stays AZ_PAD_INDEX for all 7 slots (inert).
    return out


HEX_NEIGHBORS = _build_hex_neighbors()  # (128, 7) numpy
HEX_NEIGHBORS_MX = mx.array(HEX_NEIGHBORS, dtype=mx.int32)


# ── AZD2 dataset ──

@dataclass
class AzDatasetV2:
    inputs: np.ndarray            # (N, 68, 128) — own board planes
    opp_inputs: np.ndarray        # (N, 3, 68, 128) — opponent board planes (in seat-rotation order)
    entities: np.ndarray          # (N, 10, 32) — opp slots 4..7 are zero (filled by trunk)
    tile_idx: np.ndarray          # (N, K) int32, AZ_PAD_INDEX for invalid
    wildlife_idx: np.ndarray      # (N, K) int32, -1 for skip-wildlife
    market_idx: np.ndarray        # (N, K) int32 in [0,3]
    wildlife_market_idx: np.ndarray  # (N, K) int32 in [0,3]
    mask: np.ndarray              # (N, K) bool
    policy: np.ndarray            # (N, K) float32
    value: np.ndarray             # (N,) float32 in [0,1]
    aux_values: np.ndarray        # (N, 16) float32 in [0,1]
    phase_one_hot: np.ndarray     # (N, 3) float32 (sourced from globals token)

    @property
    def size(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def max_candidates(self) -> int:
        return int(self.mask.shape[1])


def _read_u32(data: bytes, pos: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, pos)[0], pos + 4


def _read_f32(data: bytes, pos: int) -> tuple[float, int]:
    return struct.unpack_from("<f", data, pos)[0], pos + 4


def load_azd_v2(paths: list[Path], max_candidates: int | None = None) -> AzDatasetV2:
    raw_samples = []
    header = None
    observed_max = 0
    for path in paths:
        data = path.read_bytes()
        if data[:4] != AZD_MAGIC_V2:
            raise ValueError(f"{path}: bad magic {data[:4]!r}, expected {AZD_MAGIC_V2!r}")
        pos = 4
        input_channels, pos = _read_u32(data, pos)
        local_cells, pos = _read_u32(data, pos)
        cells_padded, pos = _read_u32(data, pos)
        entity_tokens, pos = _read_u32(data, pos)
        entity_raw_dim, pos = _read_u32(data, pos)
        value_subheads, pos = _read_u32(data, pos)
        max_opponents, pos = _read_u32(data, pos)
        n_samples, pos = _read_u32(data, pos)
        h = (input_channels, local_cells, cells_padded, entity_tokens,
             entity_raw_dim, value_subheads, max_opponents)
        if header is None:
            header = h
        elif header != h:
            raise ValueError(f"{path}: header mismatch {h} vs {header}")
        input_len = input_channels * cells_padded
        opp_block_len = max_opponents * input_channels * cells_padded
        entity_len = entity_tokens * entity_raw_dim
        for _ in range(n_samples):
            value, pos = _read_f32(data, pos)
            aux = np.frombuffer(data, dtype="<f4", count=value_subheads,
                                offset=pos).copy()
            pos += value_subheads * 4
            n, pos = _read_u32(data, pos)
            observed_max = max(observed_max, n)
            inp = np.frombuffer(data, dtype="<f4", count=input_len,
                                offset=pos).copy()
            pos += input_len * 4
            opp = np.frombuffer(data, dtype="<f4", count=opp_block_len,
                                offset=pos).copy()
            pos += opp_block_len * 4
            ent = np.frombuffer(data, dtype="<f4", count=entity_len,
                                offset=pos).copy()
            pos += entity_len * 4
            cands = np.frombuffer(data, dtype="<i4", count=n * 4,
                                  offset=pos).copy().reshape(n, 4)
            pos += n * 4 * 4
            policy = np.frombuffer(data, dtype="<f4", count=n,
                                   offset=pos).copy()
            pos += n * 4
            raw_samples.append((value, aux, inp, opp, ent, cands, policy))
        if pos != len(data):
            raise ValueError(f"{path}: trailing bytes ({len(data) - pos})")

    if not raw_samples:
        raise ValueError("no AZD2 samples loaded")
    if header != (AZ_INPUT_CHANNELS_V2, AZ_LOCAL_CELLS, AZ_CELLS_PADDED,
                  AZ_ENTITY_TOKENS, AZ_ENTITY_RAW_DIM, AZ_VALUE_SUBHEADS,
                  AZ_MAX_OPPONENTS):
        raise ValueError(f"AZD2 header {header} doesn't match expected v2 constants")

    k = max_candidates or observed_max
    if k < observed_max:
        raise ValueError(
            f"--max-candidates {k} smaller than observed {observed_max}"
        )

    n = len(raw_samples)
    inputs = np.zeros((n, AZ_INPUT_CHANNELS_V2, AZ_CELLS_PADDED), dtype=np.float32)
    opp_inputs = np.zeros(
        (n, AZ_MAX_OPPONENTS, AZ_INPUT_CHANNELS_V2, AZ_CELLS_PADDED),
        dtype=np.float32,
    )
    entities = np.zeros((n, AZ_ENTITY_TOKENS, AZ_ENTITY_RAW_DIM), dtype=np.float32)
    tile_idx = np.full((n, k), AZ_PAD_INDEX, dtype=np.int32)
    wildlife_idx = np.full((n, k), -1, dtype=np.int32)
    market_idx = np.zeros((n, k), dtype=np.int32)
    wildlife_market_idx = np.zeros((n, k), dtype=np.int32)
    mask = np.zeros((n, k), dtype=bool)
    policy = np.zeros((n, k), dtype=np.float32)
    values = np.zeros((n,), dtype=np.float32)
    aux_vals = np.zeros((n, AZ_VALUE_SUBHEADS), dtype=np.float32)
    phase = np.zeros((n, AZ_VALUE_PHASES), dtype=np.float32)

    for i, (value, aux, inp, opp, ent, cands, pol) in enumerate(raw_samples):
        inputs[i] = inp.reshape(AZ_INPUT_CHANNELS_V2, AZ_CELLS_PADDED)
        opp_inputs[i] = opp.reshape(
            AZ_MAX_OPPONENTS, AZ_INPUT_CHANNELS_V2, AZ_CELLS_PADDED
        )
        entities[i] = ent.reshape(AZ_ENTITY_TOKENS, AZ_ENTITY_RAW_DIM)
        m = cands.shape[0]
        # Rust writes local cell indices already; -1 for skip-wildlife.
        tile_idx[i, :m] = np.where(cands[:, 0] >= 0, cands[:, 0], AZ_PAD_INDEX)
        wildlife_idx[i, :m] = cands[:, 1]
        market_idx[i, :m] = np.clip(cands[:, 2], 0, 3)
        wildlife_market_idx[i, :m] = np.where(cands[:, 3] >= 0,
                                              cands[:, 3], cands[:, 2])
        wildlife_market_idx[i, :m] = np.clip(wildlife_market_idx[i, :m], 0, 3)
        mask[i, :m] = True
        psum = float(pol.sum())
        policy[i, :m] = pol / psum if psum > 0 else 1.0 / m
        values[i] = np.clip(value, 0.0, 1.0)
        aux_vals[i] = np.clip(aux, 0.0, 1.0)
        # Phase one-hot is already encoded at globals token (index 8), dims 0–2.
        # Rust emits the one-hot directly; copy it through.
        phase[i] = entities[i, TOKEN_GLOBALS, 0:3]
        # Sanity: the row must sum to ~1 (Rust always emits exactly one bit).
        assert abs(float(phase[i].sum()) - 1.0) < 1e-4, \
            f"phase one-hot at sample {i} sum = {float(phase[i].sum())}"

    return AzDatasetV2(
        inputs=inputs,
        opp_inputs=opp_inputs,
        entities=entities,
        tile_idx=tile_idx,
        wildlife_idx=wildlife_idx,
        market_idx=market_idx,
        wildlife_market_idx=wildlife_market_idx,
        mask=mask,
        policy=policy,
        value=values,
        aux_values=aux_vals,
        phase_one_hot=phase,
    )


# ── Initialization helpers ──

def rand_uniform(shape, scale: float):
    return mx.random.uniform(low=-scale, high=scale, shape=shape)


# ── HexConv1d: 7-tap conv over 128 cells using precomputed neighbors ──

class HexConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        scale = (2.0 / max(1, in_c * 7)) ** 0.5
        # Weight: (out_c, in_c, 7)
        self.w = rand_uniform((out_c, in_c, 7), scale)
        self.b = mx.zeros((out_c,), dtype=mx.float32)

    def __call__(self, x):
        # x: (B, in_c, 128)
        # Gather neighbors: (B, in_c, 128, 7) via mx.take along cells axis.
        gathered = mx.take(x, HEX_NEIGHBORS_MX, axis=2)  # (B, in_c, 128, 7)
        # Move 128 to the front of the inner product axes.
        # gathered: (B, in_c, 128, 7) → (B, 128, in_c * 7)
        B = x.shape[0]
        g = mx.transpose(gathered, (0, 2, 1, 3))
        g = g.reshape(B, AZ_CELLS_PADDED, self.in_c * 7)
        w_flat = self.w.reshape(self.out_c, self.in_c * 7)
        # (B, 128, out_c) = g @ w_flat.T
        out = g @ mx.transpose(w_flat)
        # → (B, out_c, 128)
        out = mx.transpose(out, (0, 2, 1))
        out = out + self.b.reshape(1, self.out_c, 1)
        return out


class ResHexBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = HexConv(channels, channels)
        self.c2 = HexConv(channels, channels)

    def __call__(self, x):
        z1 = mx.maximum(self.c1(x), 0.0)
        z2 = self.c2(z1)
        return mx.maximum(z2 + x, 0.0)


# ── LayerNorm (over the last dim) ──

def layer_norm(x: mx.array, scale: mx.array, bias: mx.array) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + LN_EPS) * scale + bias


# ── Set-Transformer SAB block (pre-norm) ──

class Sab(nn.Module):
    def __init__(self, d: int, heads: int, ffn_dim: int):
        super().__init__()
        assert d % heads == 0
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.ffn_dim = ffn_dim
        self.ln1_scale = mx.ones((d,), dtype=mx.float32)
        self.ln1_bias = mx.zeros((d,), dtype=mx.float32)
        s_attn = (2.0 / d) ** 0.5
        self.qkv_w = rand_uniform((3 * d, d), s_attn)
        self.qkv_b = mx.zeros((3 * d,), dtype=mx.float32)
        self.out_w = rand_uniform((d, d), s_attn)
        self.out_b = mx.zeros((d,), dtype=mx.float32)
        self.ln2_scale = mx.ones((d,), dtype=mx.float32)
        self.ln2_bias = mx.zeros((d,), dtype=mx.float32)
        s_ffn = (2.0 / d) ** 0.5
        self.ffn1_w = rand_uniform((ffn_dim, d), s_ffn)
        self.ffn1_b = mx.zeros((ffn_dim,), dtype=mx.float32)
        self.ffn2_w = rand_uniform((d, ffn_dim), (2.0 / ffn_dim) ** 0.5)
        self.ffn2_b = mx.zeros((d,), dtype=mx.float32)

    def __call__(self, x):
        # x: (B, N, d)
        B, N, d = x.shape
        # Pre-norm 1.
        x_ln = layer_norm(x, self.ln1_scale, self.ln1_bias)
        # QKV.
        qkv = x_ln @ mx.transpose(self.qkv_w) + self.qkv_b  # (B, N, 3d)
        q = qkv[:, :, 0 * d : 1 * d]
        k = qkv[:, :, 1 * d : 2 * d]
        v = qkv[:, :, 2 * d : 3 * d]
        # Reshape for multi-head: (B, N, heads, head_dim) → (B, heads, N, head_dim)
        def split(z):
            return mx.transpose(z.reshape(B, N, self.heads, self.head_dim), (0, 2, 1, 3))
        qh, kh, vh = split(q), split(k), split(v)
        scale = 1.0 / (self.head_dim ** 0.5)
        # Attention scores: (B, heads, N, N)
        scores = qh @ mx.transpose(kh, (0, 1, 3, 2)) * scale
        attn = mx.softmax(scores, axis=-1)
        ctx = attn @ vh  # (B, heads, N, head_dim)
        # Merge heads.
        ctx = mx.transpose(ctx, (0, 2, 1, 3)).reshape(B, N, d)
        # Output projection + residual on x (not x_ln).
        after_attn = ctx @ mx.transpose(self.out_w) + self.out_b + x
        # Pre-norm 2 + FFN + residual.
        x2 = layer_norm(after_attn, self.ln2_scale, self.ln2_bias)
        h = mx.maximum(x2 @ mx.transpose(self.ffn1_w) + self.ffn1_b, 0.0)
        out = h @ mx.transpose(self.ffn2_w) + self.ffn2_b + after_attn
        return out


# ── Cross-Attention: Q from per-cell trunk, K/V from entity tokens ──

class CrossAttn(nn.Module):
    def __init__(self, trunk_c: int, d_ent: int, heads: int):
        super().__init__()
        assert d_ent % heads == 0
        self.trunk_c = trunk_c
        self.d_ent = d_ent
        self.heads = heads
        self.head_dim = d_ent // heads
        s_q = (2.0 / trunk_c) ** 0.5
        s_kv = (2.0 / d_ent) ** 0.5
        s_out = (2.0 / d_ent) ** 0.5
        self.q_w = rand_uniform((d_ent, trunk_c), s_q)
        self.q_b = mx.zeros((d_ent,), dtype=mx.float32)
        self.k_w = rand_uniform((d_ent, d_ent), s_kv)
        self.k_b = mx.zeros((d_ent,), dtype=mx.float32)
        self.v_w = rand_uniform((d_ent, d_ent), s_kv)
        self.v_b = mx.zeros((d_ent,), dtype=mx.float32)
        self.out_w = rand_uniform((trunk_c, d_ent), s_out)
        self.out_b = mx.zeros((trunk_c,), dtype=mx.float32)

    def __call__(self, trunk, entities):
        # trunk: (B, trunk_c, 128), entities: (B, N_tok, d_ent)
        B = trunk.shape[0]
        N_cells = trunk.shape[2]
        N_tok = entities.shape[1]
        d = self.d_ent
        # Project Q from trunk: (B, 128, d_ent)
        trunk_cells = mx.transpose(trunk, (0, 2, 1))  # (B, 128, trunk_c)
        q = trunk_cells @ mx.transpose(self.q_w) + self.q_b  # (B, 128, d)
        # Project K, V from entities: (B, N_tok, d)
        k = entities @ mx.transpose(self.k_w) + self.k_b
        v = entities @ mx.transpose(self.v_w) + self.v_b
        # Reshape for heads: (B, X, heads, head_dim) → (B, heads, X, head_dim)
        def split(z, X):
            return mx.transpose(z.reshape(B, X, self.heads, self.head_dim),
                                (0, 2, 1, 3))
        qh = split(q, N_cells)
        kh = split(k, N_tok)
        vh = split(v, N_tok)
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = qh @ mx.transpose(kh, (0, 1, 3, 2)) * scale  # (B, h, 128, N_tok)
        attn = mx.softmax(scores, axis=-1)
        ctx = attn @ vh  # (B, heads, 128, head_dim)
        ctx = mx.transpose(ctx, (0, 2, 1, 3)).reshape(B, N_cells, d)
        # Project back to trunk channels.
        delta = ctx @ mx.transpose(self.out_w) + self.out_b  # (B, 128, trunk_c)
        delta = mx.transpose(delta, (0, 2, 1))  # (B, trunk_c, 128)
        return mx.maximum(trunk + delta, 0.0)


# ── Multi-Head Value with phase gate ──

class MultiHeadValue(nn.Module):
    def __init__(self, channels: int, hidden: int, subheads: int):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.subheads = subheads
        s1 = (2.0 / channels) ** 0.5
        s2 = (2.0 / hidden) ** 0.5
        self.w1 = rand_uniform((hidden, channels), s1)
        self.b1 = mx.zeros((hidden,), dtype=mx.float32)
        self.w2 = rand_uniform((subheads, hidden), s2)
        self.b2 = mx.zeros((subheads,), dtype=mx.float32)
        # Blend matrix: (3 phases, subheads). Init to zeros (uniform softmax).
        self.blend = mx.zeros((AZ_VALUE_PHASES, subheads), dtype=mx.float32)

    def __call__(self, pooled, phase):
        # pooled: (B, channels), phase: (B, 3)
        h = mx.maximum(pooled @ mx.transpose(self.w1) + self.b1, 0.0)
        subs = mx.sigmoid(h @ mx.transpose(self.w2) + self.b2)  # (B, subheads)
        # Blend weights per phase, softmaxed across subheads.
        phase_logits = phase @ self.blend  # (B, subheads)
        # Softmax across subheads for hard-phase samples this is identical to
        # softmax(self.blend[phase_idx, :]); for soft phases it's a linear
        # combination of phase logits, then softmax.
        weights = mx.softmax(phase_logits, axis=-1)
        v = mx.sum(weights * subs, axis=-1)  # (B,)
        return v, subs


# ── Full CascadiaAzNetV2 ──

class CascadiaAzNetV2(nn.Module):
    def __init__(
        self,
        channels: int,
        blocks: int,
        entity_dim: int,
        sab_blocks: int,
        heads: int,
        value_hidden: int,
        max_candidates: int,
        c_puct: float,
    ):
        super().__init__()
        self.channels = channels
        self.blocks_count = blocks
        self.entity_dim = entity_dim
        self.sab_blocks_count = sab_blocks
        self.heads = heads
        self.value_hidden = value_hidden
        self.value_subheads = AZ_VALUE_SUBHEADS
        self.max_candidates = max_candidates
        self.c_puct = c_puct

        self.stem = HexConv(AZ_INPUT_CHANNELS_V2, channels)
        self.blocks = [ResHexBlock(channels) for _ in range(blocks)]

        # Shared opponent trunk — applied to each of AZ_MAX_OPPONENTS boards
        # with the same weights. Output is pooled to 32-dim and slotted into
        # entity tokens 4..7 before the entity up-projection.
        self.opp_stem = HexConv(AZ_INPUT_CHANNELS_V2, AZ_OPP_TRUNK_CHANNELS)
        self.opp_blocks = [
            ResHexBlock(AZ_OPP_TRUNK_CHANNELS) for _ in range(AZ_OPP_TRUNK_BLOCKS)
        ]

        # Entity stream: 32 → entity_dim, then SABs.
        s_up = (2.0 / AZ_ENTITY_RAW_DIM) ** 0.5
        self.ent_up_w = rand_uniform((entity_dim, AZ_ENTITY_RAW_DIM), s_up)
        self.ent_up_b = mx.zeros((entity_dim,), dtype=mx.float32)
        self.sabs = [Sab(entity_dim, heads, AZ_SAB_FFN_DIM) for _ in range(sab_blocks)]

        self.cross = CrossAttn(channels, entity_dim, AZ_CROSS_HEADS)

        # Factorized policy heads.
        head_scale = (2.0 / channels) ** 0.5
        market_scale = (2.0 / (channels + entity_dim)) ** 0.5
        self.policy_tile_w = rand_uniform((channels,), head_scale)
        self.policy_tile_b = mx.array(0.0, dtype=mx.float32)
        self.policy_wildlife_w = rand_uniform((channels,), head_scale)
        self.policy_wildlife_b = mx.array(0.0, dtype=mx.float32)
        self.policy_market_w = rand_uniform((4, channels + entity_dim), market_scale)
        self.policy_market_b = mx.zeros((4,), dtype=mx.float32)
        self.policy_wildlife_market_w = rand_uniform((4, channels + entity_dim),
                                                     market_scale)
        self.policy_wildlife_market_b = mx.zeros((4,), dtype=mx.float32)
        self.policy_skip_w = rand_uniform((channels,), head_scale)
        self.policy_skip_b = mx.array(0.0, dtype=mx.float32)

        # Value head.
        self.value_head = MultiHeadValue(channels, value_hidden, AZ_VALUE_SUBHEADS)

    def trunk_forward(self, x):
        # x: (B, 68, 128)
        x = mx.maximum(self.stem(x), 0.0)
        for blk in self.blocks:
            x = blk(x)
        return x

    def encode_opps(self, opp_inputs):
        """Run the shared opp trunk on each opponent board and pool.

        opp_inputs: (B, 3, 68, 128) — 3 opponent boards per sample.
        Returns:    (B, 3, AZ_OPP_TRUNK_CHANNELS) pooled vectors.
        """
        B, K, _, _ = opp_inputs.shape
        # Flatten the opp axis so the shared trunk processes all opps at once.
        x = opp_inputs.reshape(B * K, AZ_INPUT_CHANNELS_V2, AZ_CELLS_PADDED)
        x = mx.maximum(self.opp_stem(x), 0.0)
        for blk in self.opp_blocks:
            x = blk(x)
        # x: (B*K, AZ_OPP_TRUNK_CHANNELS, 128). Mean-pool over real cells.
        real = x[:, :, :AZ_LOCAL_CELLS]
        pooled = mx.mean(real, axis=2)  # (B*K, AZ_OPP_TRUNK_CHANNELS)
        return pooled.reshape(B, K, AZ_OPP_TRUNK_CHANNELS)

    def entity_forward(self, ent_raw):
        # ent_raw: (B, 10, 32) → project to (B, 10, d) → SABs.
        ent = ent_raw @ mx.transpose(self.ent_up_w) + self.ent_up_b
        for sab in self.sabs:
            ent = sab(ent)
        return ent

    def __call__(self, batch):
        x = self.trunk_forward(batch["inputs"])
        # Shared opp trunk: pooled vectors slot into entity tokens 4..7.
        opp_pooled = self.encode_opps(batch["opp_inputs"])  # (B, 3, 32)
        ent_raw = batch["entities"]
        # Replace dims 0..AZ_OPP_TRUNK_CHANNELS at the opp slots. Because
        # AZ_OPP_TRUNK_CHANNELS == AZ_ENTITY_RAW_DIM, we overwrite the whole
        # token. Use index_update via concatenation: build a new tensor.
        prefix = ent_raw[:, :4, :]                  # (B, 4, 32)  market slots
        suffix = ent_raw[:, 4 + AZ_MAX_OPPONENTS:, :]  # (B, 3, 32) bag/globals/race
        ent_raw_with_opps = mx.concatenate([prefix, opp_pooled, suffix], axis=1)
        ent = self.entity_forward(ent_raw_with_opps)
        fused = self.cross(x, ent)  # (B, channels, 128)

        # Pool over real cells (exclude pad index AZ_PAD_INDEX = 127).
        real = fused[:, :, :AZ_LOCAL_CELLS]
        pooled = mx.mean(real, axis=2)  # (B, channels)

        # Value.
        value, sub_pred = self.value_head(pooled, batch["phase"])

        # Policy logits.
        tile_logits = mx.einsum("bch,c->bh", fused, self.policy_tile_w) + self.policy_tile_b
        wildlife_logits = (
            mx.einsum("bch,c->bh", fused, self.policy_wildlife_w) + self.policy_wildlife_b
        )

        # Market and wildlife-market: per slot, concat(pooled, ent_slot).
        ent_market = ent[:, :4, :]  # (B, 4, d)
        # Build (B, 4, channels + d) by broadcasting pooled.
        pooled_b = mx.broadcast_to(pooled[:, None, :],
                                   (pooled.shape[0], 4, pooled.shape[1]))
        concat = mx.concatenate([pooled_b, ent_market], axis=-1)  # (B, 4, c+d)
        # Per-slot dot with policy_market_w[slot, :].
        market_logits = mx.sum(concat * self.policy_market_w[None, :, :], axis=-1) + self.policy_market_b
        wildlife_market_logits = (
            mx.sum(concat * self.policy_wildlife_market_w[None, :, :], axis=-1)
            + self.policy_wildlife_market_b
        )
        skip_logits = pooled @ self.policy_skip_w + self.policy_skip_b  # (B,)

        # Assemble per-candidate logits.
        tile = mx.take_along_axis(tile_logits, batch["tile_idx"], axis=1)
        safe_wildlife_idx = mx.maximum(batch["wildlife_idx"], 0)
        wildlife = mx.where(
            batch["wildlife_idx"] >= 0,
            mx.take_along_axis(wildlife_logits, safe_wildlife_idx, axis=1),
            skip_logits[:, None],
        )
        market = mx.take_along_axis(market_logits, batch["market_idx"], axis=1)
        wildlife_market = mx.take_along_axis(
            wildlife_market_logits, batch["wildlife_market_idx"], axis=1
        )
        logits = tile + wildlife + market + wildlife_market
        return logits, value, sub_pred


# ── Loss + training loop ──

def make_batch(ds: AzDatasetV2, idx: np.ndarray) -> dict[str, mx.array]:
    return {
        "inputs": mx.array(ds.inputs[idx]),
        "opp_inputs": mx.array(ds.opp_inputs[idx]),
        "entities": mx.array(ds.entities[idx]),
        "tile_idx": mx.array(ds.tile_idx[idx]),
        "wildlife_idx": mx.array(ds.wildlife_idx[idx]),
        "market_idx": mx.array(ds.market_idx[idx]),
        "wildlife_market_idx": mx.array(ds.wildlife_market_idx[idx]),
        "mask": mx.array(ds.mask[idx]),
        "policy": mx.array(ds.policy[idx]),
        "value": mx.array(ds.value[idx]),
        "aux_values": mx.array(ds.aux_values[idx]),
        "phase": mx.array(ds.phase_one_hot[idx]),
    }


def loss_fn(model: CascadiaAzNetV2, batch: dict[str, mx.array],
            value_weight: float, aux_weight: float):
    logits, value, sub_pred = model(batch)
    neg_inf = mx.array(-1.0e9, dtype=mx.float32)
    masked = mx.where(batch["mask"], logits, neg_inf)
    log_probs = masked - mx.logsumexp(masked, axis=1, keepdims=True)
    policy_loss = -mx.sum(batch["policy"] * log_probs, axis=1).mean()
    value_loss = mx.mean(mx.square(value - batch["value"]))
    aux_loss = mx.mean(mx.square(sub_pred - batch["aux_values"]))
    return policy_loss + value_weight * value_loss + aux_weight * aux_loss


def eval_losses(model: CascadiaAzNetV2, ds: AzDatasetV2, batch_size: int,
                value_weight: float, aux_weight: float):
    total = 0.0
    top1 = 0
    seen = 0
    for start in range(0, ds.size, batch_size):
        idx = np.arange(start, min(ds.size, start + batch_size))
        batch = make_batch(ds, idx)
        loss = loss_fn(model, batch, value_weight, aux_weight)
        logits, _v, _s = model(batch)
        masked = mx.where(batch["mask"], logits, mx.array(-1.0e9, dtype=mx.float32))
        pred = np.array(mx.argmax(masked, axis=1))
        target = np.array(np.argmax(ds.policy[idx], axis=1))
        top1 += int((pred == target).sum())
        seen += len(idx)
        total += float(np.array(loss)) * len(idx)
    return total / max(1, seen), top1 / max(1, seen)


def _materialize_dataset_to_mx(ds: AzDatasetV2) -> dict[str, mx.array]:
    """Upload the whole training set to MLX (Apple GPU) memory once. Eliminates
    the per-step numpy→mx.array copy. Sliding-window datasets at our scale
    (~50–100 K samples × ~9.5 KB/sample ≈ ~1 GB) fit easily on M-series Macs."""
    return {
        "inputs": mx.array(ds.inputs),
        "opp_inputs": mx.array(ds.opp_inputs),
        "entities": mx.array(ds.entities),
        "tile_idx": mx.array(ds.tile_idx),
        "wildlife_idx": mx.array(ds.wildlife_idx),
        "market_idx": mx.array(ds.market_idx),
        "wildlife_market_idx": mx.array(ds.wildlife_market_idx),
        "mask": mx.array(ds.mask),
        "policy": mx.array(ds.policy),
        "value": mx.array(ds.value),
        "aux_values": mx.array(ds.aux_values),
        "phase": mx.array(ds.phase_one_hot),
    }


def _slice_mx(ds_mx: dict[str, mx.array], idx_mx: mx.array) -> dict[str, mx.array]:
    """Gather a batch from the on-GPU dataset. `idx_mx` is a 1-D int32 array
    of sample indices; we `mx.take` along axis 0 for every field."""
    return {k: mx.take(v, idx_mx, axis=0) for k, v in ds_mx.items()}


def train(model: CascadiaAzNetV2, train_ds: AzDatasetV2,
          val_ds: AzDatasetV2 | None, args):
    """Phase 0.8.E + perf-pass MLX training:

    1. Whole dataset uploaded to GPU once (`_materialize_dataset_to_mx`).
    2. Per-batch indexing via `mx.take` on the persistent array — no
       numpy→mx.array conversion in the hot path.
    3. The (loss, grads) computation + optimizer step is wrapped in
       `mx.compile`, which traces the graph on first call and reuses the
       compiled kernel on subsequent batches. Real MLX users routinely see
       2–5× from this on small/medium nets.
    4. Default batch size doubled (128 → 256) — fills more AMX tiles per
       SGEMM and amortizes per-step overhead.

    Strictly behavior-preserving: same math, same loss, same optimizer
    semantics; only the *execution graph* is materialized differently.
    """
    opt = optim.Adam(learning_rate=args.lr)

    # Persistent on-GPU dataset (one numpy→mx.array copy per epoch's worth
    # of batches instead of one copy per batch).
    train_mx = _materialize_dataset_to_mx(train_ds)
    val_mx = _materialize_dataset_to_mx(val_ds) if val_ds is not None and val_ds.size > 0 else None

    loss_and_grad = nn.value_and_grad(
        model,
        lambda m, b: loss_fn(m, b, args.value_weight, args.aux_weight),
    )

    # `mx.compile` traces the step function on first call and caches the
    # compiled graph. Subsequent calls skip Python-level op construction.
    # The `inputs=` / `outputs=` kwargs declare the mutable state MLX must
    # capture (model params + optimizer slots + the global RNG state).
    state = [model.state, opt.state, mx.random.state]

    def _step(batch):
        loss, grads = loss_and_grad(model, batch)
        opt.update(model, grads)
        return loss

    step = mx.compile(_step, inputs=state, outputs=state)

    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(train_ds.size)
        running = 0.0
        seen = 0
        for start in range(0, train_ds.size, args.batch_size):
            np_idx = order[start : start + args.batch_size]
            idx_mx = mx.array(np_idx.astype(np.int32))
            batch = _slice_mx(train_mx, idx_mx)
            loss = step(batch)
            mx.eval(state)
            running += float(loss) * len(np_idx)
            seen += len(np_idx)
        train_loss = running / max(1, seen)
        msg = f"epoch {epoch:03d}: train_loss={train_loss:.5f} elapsed={time.time() - t0:.1f}s"
        if val_mx is not None:
            val_loss, top1 = _eval_losses_mx(
                model, val_ds, val_mx, args.batch_size,
                args.value_weight, args.aux_weight,
            )
            msg += f" val_loss={val_loss:.5f} val_top1={top1:.3f}"
        print(msg, flush=True)
        if args.save_each_epoch:
            stem = Path(args.out).with_suffix("").as_posix()
            save_azr_v2(model, Path(f"{stem}_epoch{epoch}.azr"))
    return model


def _eval_losses_mx(model, val_ds, val_mx, batch_size, value_weight, aux_weight):
    """Eval loop using the persistent on-GPU val set. Mirrors the original
    `eval_losses` but reads from `val_mx` instead of going through numpy.
    """
    total = 0.0
    top1 = 0
    seen = 0
    n = val_ds.size
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        np_idx = np.arange(start, end, dtype=np.int32)
        idx_mx = mx.array(np_idx)
        batch = _slice_mx(val_mx, idx_mx)
        loss = loss_fn(model, batch, value_weight, aux_weight)
        logits, _v, _s = model(batch)
        masked = mx.where(batch["mask"], logits, mx.array(-1.0e9, dtype=mx.float32))
        pred = np.array(mx.argmax(masked, axis=1))
        target = np.array(np.argmax(val_ds.policy[start:end], axis=1))
        top1 += int((pred == target).sum())
        seen += (end - start)
        total += float(loss) * (end - start)
    return total / max(1, seen), top1 / max(1, seen)


# ── AZR2 save/load (must match Rust AlphaZeroNetV2::{save, load}) ──

def _write_u32(f, v: int):
    f.write(struct.pack("<I", int(v)))


def _write_f32(f, v: float):
    f.write(struct.pack("<f", float(v)))


def _write_vec(f, arr):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    _write_u32(f, flat.size)
    f.write(flat.astype("<f4", copy=False).tobytes())


def _as_np(x, shape=None):
    arr = np.array(x, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def save_conv(f, conv: HexConv):
    _write_u32(f, conv.in_c)
    _write_u32(f, conv.out_c)
    _write_vec(f, _as_np(conv.w))
    _write_vec(f, _as_np(conv.b))


def save_sab(f, sab: Sab):
    _write_vec(f, _as_np(sab.ln1_scale))
    _write_vec(f, _as_np(sab.ln1_bias))
    _write_vec(f, _as_np(sab.qkv_w))
    _write_vec(f, _as_np(sab.qkv_b))
    _write_vec(f, _as_np(sab.out_w))
    _write_vec(f, _as_np(sab.out_b))
    _write_vec(f, _as_np(sab.ln2_scale))
    _write_vec(f, _as_np(sab.ln2_bias))
    _write_vec(f, _as_np(sab.ffn1_w))
    _write_vec(f, _as_np(sab.ffn1_b))
    _write_vec(f, _as_np(sab.ffn2_w))
    _write_vec(f, _as_np(sab.ffn2_b))


def save_cross(f, cross: CrossAttn):
    _write_vec(f, _as_np(cross.q_w))
    _write_vec(f, _as_np(cross.q_b))
    _write_vec(f, _as_np(cross.k_w))
    _write_vec(f, _as_np(cross.k_b))
    _write_vec(f, _as_np(cross.v_w))
    _write_vec(f, _as_np(cross.v_b))
    _write_vec(f, _as_np(cross.out_w))
    _write_vec(f, _as_np(cross.out_b))


def save_azr_v2(model: CascadiaAzNetV2, path: Path):
    mx.eval(model.parameters())
    with path.open("wb") as f:
        f.write(AZR_MAGIC_V2)
        _write_u32(f, model.channels)
        _write_u32(f, model.blocks_count)
        _write_u32(f, model.entity_dim)
        _write_u32(f, model.sab_blocks_count)
        _write_u32(f, model.heads)
        _write_u32(f, model.value_hidden)
        _write_u32(f, model.value_subheads)
        _write_u32(f, model.max_candidates)
        _write_f32(f, model.c_puct)
        # Opp-trunk header (constants; recorded for explicit version safety).
        _write_u32(f, AZ_OPP_TRUNK_CHANNELS)
        _write_u32(f, AZ_OPP_TRUNK_BLOCKS)
        _write_u32(f, AZ_MAX_OPPONENTS)
        # Main trunk.
        save_conv(f, model.stem)
        for blk in model.blocks:
            save_conv(f, blk.c1)
            save_conv(f, blk.c2)
        # Shared opp trunk.
        save_conv(f, model.opp_stem)
        for blk in model.opp_blocks:
            save_conv(f, blk.c1)
            save_conv(f, blk.c2)
        _write_vec(f, _as_np(model.ent_up_w))
        _write_vec(f, _as_np(model.ent_up_b))
        for sab in model.sabs:
            save_sab(f, sab)
        save_cross(f, model.cross)
        _write_vec(f, _as_np(model.policy_tile_w))
        _write_f32(f, float(np.array(model.policy_tile_b)))
        _write_vec(f, _as_np(model.policy_wildlife_w))
        _write_f32(f, float(np.array(model.policy_wildlife_b)))
        _write_vec(f, _as_np(model.policy_market_w))
        for v in _as_np(model.policy_market_b):
            _write_f32(f, float(v))
        _write_vec(f, _as_np(model.policy_wildlife_market_w))
        for v in _as_np(model.policy_wildlife_market_b):
            _write_f32(f, float(v))
        _write_vec(f, _as_np(model.policy_skip_w))
        _write_f32(f, float(np.array(model.policy_skip_b)))
        _write_vec(f, _as_np(model.value_head.w1))
        _write_vec(f, _as_np(model.value_head.b1))
        _write_vec(f, _as_np(model.value_head.w2))
        _write_vec(f, _as_np(model.value_head.b2))
        _write_vec(f, _as_np(model.value_head.blend))
    print(f"saved {path}", flush=True)


def _read_vec(data: bytes, pos: int) -> tuple[np.ndarray, int]:
    n, pos = _read_u32(data, pos)
    arr = np.frombuffer(data, dtype="<f4", count=n, offset=pos).copy()
    return arr, pos + n * 4


def _load_conv(data: bytes, pos: int, conv: HexConv) -> int:
    in_c, pos = _read_u32(data, pos)
    out_c, pos = _read_u32(data, pos)
    if (in_c, out_c) != (conv.in_c, conv.out_c):
        raise ValueError(f"conv shape mismatch: {(in_c, out_c)} vs {(conv.in_c, conv.out_c)}")
    w, pos = _read_vec(data, pos)
    b, pos = _read_vec(data, pos)
    conv.w = mx.array(w.reshape(out_c, in_c, 7))
    conv.b = mx.array(b.reshape(out_c))
    return pos


def _load_sab(data: bytes, pos: int, sab: Sab) -> int:
    d = sab.d
    ffn_dim = sab.ffn_dim
    for name in ("ln1_scale", "ln1_bias"):
        v, pos = _read_vec(data, pos)
        setattr(sab, name, mx.array(v.reshape(d)))
    v, pos = _read_vec(data, pos)
    sab.qkv_w = mx.array(v.reshape(3 * d, d))
    v, pos = _read_vec(data, pos)
    sab.qkv_b = mx.array(v.reshape(3 * d))
    v, pos = _read_vec(data, pos)
    sab.out_w = mx.array(v.reshape(d, d))
    v, pos = _read_vec(data, pos)
    sab.out_b = mx.array(v.reshape(d))
    for name in ("ln2_scale", "ln2_bias"):
        v, pos = _read_vec(data, pos)
        setattr(sab, name, mx.array(v.reshape(d)))
    v, pos = _read_vec(data, pos)
    sab.ffn1_w = mx.array(v.reshape(ffn_dim, d))
    v, pos = _read_vec(data, pos)
    sab.ffn1_b = mx.array(v.reshape(ffn_dim))
    v, pos = _read_vec(data, pos)
    sab.ffn2_w = mx.array(v.reshape(d, ffn_dim))
    v, pos = _read_vec(data, pos)
    sab.ffn2_b = mx.array(v.reshape(d))
    return pos


def _load_cross(data: bytes, pos: int, cross: CrossAttn) -> int:
    d = cross.d_ent
    c = cross.trunk_c
    v, pos = _read_vec(data, pos); cross.q_w = mx.array(v.reshape(d, c))
    v, pos = _read_vec(data, pos); cross.q_b = mx.array(v.reshape(d))
    v, pos = _read_vec(data, pos); cross.k_w = mx.array(v.reshape(d, d))
    v, pos = _read_vec(data, pos); cross.k_b = mx.array(v.reshape(d))
    v, pos = _read_vec(data, pos); cross.v_w = mx.array(v.reshape(d, d))
    v, pos = _read_vec(data, pos); cross.v_b = mx.array(v.reshape(d))
    v, pos = _read_vec(data, pos); cross.out_w = mx.array(v.reshape(c, d))
    v, pos = _read_vec(data, pos); cross.out_b = mx.array(v.reshape(c))
    return pos


def load_azr_v2(path: Path) -> CascadiaAzNetV2:
    data = path.read_bytes()
    if data[:4] != AZR_MAGIC_V2:
        raise ValueError(f"{path}: bad magic {data[:4]!r}, expected {AZR_MAGIC_V2!r}")
    pos = 4
    channels, pos = _read_u32(data, pos)
    blocks, pos = _read_u32(data, pos)
    entity_dim, pos = _read_u32(data, pos)
    sab_blocks, pos = _read_u32(data, pos)
    heads, pos = _read_u32(data, pos)
    value_hidden, pos = _read_u32(data, pos)
    value_subheads, pos = _read_u32(data, pos)
    max_candidates, pos = _read_u32(data, pos)
    c_puct, pos = _read_f32(data, pos)
    opp_channels, pos = _read_u32(data, pos)
    opp_blocks_n, pos = _read_u32(data, pos)
    max_opp, pos = _read_u32(data, pos)
    if (opp_channels, opp_blocks_n, max_opp) != (
        AZ_OPP_TRUNK_CHANNELS, AZ_OPP_TRUNK_BLOCKS, AZ_MAX_OPPONENTS,
    ):
        raise ValueError(
            f"AZR2 opp-trunk header mismatch: got channels={opp_channels} "
            f"blocks={opp_blocks_n} max_opp={max_opp}, expected "
            f"{AZ_OPP_TRUNK_CHANNELS}/{AZ_OPP_TRUNK_BLOCKS}/{AZ_MAX_OPPONENTS}"
        )
    model = CascadiaAzNetV2(channels, blocks, entity_dim, sab_blocks, heads,
                            value_hidden, max_candidates, c_puct)
    pos = _load_conv(data, pos, model.stem)
    for blk in model.blocks:
        pos = _load_conv(data, pos, blk.c1)
        pos = _load_conv(data, pos, blk.c2)
    pos = _load_conv(data, pos, model.opp_stem)
    for blk in model.opp_blocks:
        pos = _load_conv(data, pos, blk.c1)
        pos = _load_conv(data, pos, blk.c2)
    v, pos = _read_vec(data, pos)
    model.ent_up_w = mx.array(v.reshape(entity_dim, AZ_ENTITY_RAW_DIM))
    v, pos = _read_vec(data, pos)
    model.ent_up_b = mx.array(v.reshape(entity_dim))
    for sab in model.sabs:
        pos = _load_sab(data, pos, sab)
    pos = _load_cross(data, pos, model.cross)

    v, pos = _read_vec(data, pos); model.policy_tile_w = mx.array(v.reshape(channels))
    model.policy_tile_b = mx.array(_read_f32(data, pos)[0]); pos += 4
    v, pos = _read_vec(data, pos); model.policy_wildlife_w = mx.array(v.reshape(channels))
    model.policy_wildlife_b = mx.array(_read_f32(data, pos)[0]); pos += 4
    v, pos = _read_vec(data, pos)
    model.policy_market_w = mx.array(v.reshape(4, channels + entity_dim))
    b = []
    for _ in range(4):
        x, pos = _read_f32(data, pos); b.append(x)
    model.policy_market_b = mx.array(np.array(b, dtype=np.float32))
    v, pos = _read_vec(data, pos)
    model.policy_wildlife_market_w = mx.array(v.reshape(4, channels + entity_dim))
    b = []
    for _ in range(4):
        x, pos = _read_f32(data, pos); b.append(x)
    model.policy_wildlife_market_b = mx.array(np.array(b, dtype=np.float32))
    v, pos = _read_vec(data, pos); model.policy_skip_w = mx.array(v.reshape(channels))
    model.policy_skip_b = mx.array(_read_f32(data, pos)[0]); pos += 4

    v, pos = _read_vec(data, pos)
    model.value_head.w1 = mx.array(v.reshape(value_hidden, channels))
    v, pos = _read_vec(data, pos)
    model.value_head.b1 = mx.array(v.reshape(value_hidden))
    v, pos = _read_vec(data, pos)
    model.value_head.w2 = mx.array(v.reshape(value_subheads, value_hidden))
    v, pos = _read_vec(data, pos)
    model.value_head.b2 = mx.array(v.reshape(value_subheads))
    v, pos = _read_vec(data, pos)
    model.value_head.blend = mx.array(v.reshape(AZ_VALUE_PHASES, value_subheads))

    if pos != len(data):
        raise ValueError(f"{path}: trailing bytes {len(data) - pos}")
    return model


def split_dataset(ds: AzDatasetV2, val_fraction: float, seed: int):
    if val_fraction <= 0 or ds.size < 2:
        return ds, None
    rng = np.random.default_rng(seed)
    order = rng.permutation(ds.size)
    n_val = max(1, int(round(ds.size * val_fraction)))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    if train_idx.size == 0:
        return ds, None

    def take(idx):
        return AzDatasetV2(
            inputs=ds.inputs[idx],
            opp_inputs=ds.opp_inputs[idx],
            entities=ds.entities[idx],
            tile_idx=ds.tile_idx[idx],
            wildlife_idx=ds.wildlife_idx[idx],
            market_idx=ds.market_idx[idx],
            wildlife_market_idx=ds.wildlife_market_idx[idx],
            mask=ds.mask[idx],
            policy=ds.policy[idx],
            value=ds.value[idx],
            aux_values=ds.aux_values[idx],
            phase_one_hot=ds.phase_one_hot[idx],
        )

    return take(train_idx), take(val_idx)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", nargs="+", required=True,
                   help="One or more AZD2 sample files")
    p.add_argument("--out", required=True, help="Output AZR2 weights")
    p.add_argument("--init", help="Optional AZR2 checkpoint to resume from")
    p.add_argument("--channels", type=int, default=AZ_TRUNK_CHANNELS_DEFAULT)
    p.add_argument("--blocks", type=int, default=AZ_TRUNK_BLOCKS_DEFAULT)
    p.add_argument("--entity-dim", type=int, default=AZ_ENTITY_DIM_DEFAULT)
    p.add_argument("--sab", type=int, default=AZ_SAB_BLOCKS_DEFAULT,
                   help="Number of Set-Transformer SAB blocks")
    p.add_argument("--heads", type=int, default=AZ_ATTN_HEADS_DEFAULT)
    p.add_argument("--hidden", type=int, default=AZ_VALUE_HIDDEN_DEFAULT,
                   help="Value-head hidden width")
    p.add_argument("--max-candidates", type=int)
    p.add_argument("--c-puct", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05,
                   help="(Reserved) fraction of steps for LR warmup")
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-each-epoch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    mx.random.seed(args.seed)
    paths = [Path(p) for p in args.samples]
    ds = load_azd_v2(paths, args.max_candidates)
    print(
        f"MLX device={mx.default_device()} samples={ds.size} "
        f"channels={AZ_INPUT_CHANNELS_V2} local_cells={AZ_LOCAL_CELLS} "
        f"max_candidates={ds.max_candidates}",
        flush=True,
    )
    train_ds, val_ds = split_dataset(ds, args.val_fraction, args.seed)
    if args.init:
        model = load_azr_v2(Path(args.init))
        if model.max_candidates != ds.max_candidates:
            print(
                f"warning: init max_candidates={model.max_candidates}, "
                f"data max_candidates={ds.max_candidates}; updating header",
                flush=True,
            )
            model.max_candidates = ds.max_candidates
    else:
        model = CascadiaAzNetV2(
            args.channels, args.blocks, args.entity_dim, args.sab,
            args.heads, args.hidden, ds.max_candidates, args.c_puct,
        )
    train(model, train_ds, val_ds, args)
    save_azr_v2(model, Path(args.out))


if __name__ == "__main__":
    main()
