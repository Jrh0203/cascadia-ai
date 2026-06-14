#!/usr/bin/env python3
"""Train HybridNet's Δ head on residual labels.

Reads a HYBR file (16-plane board + nnue_pred + label per record). Trains a
HexCNN matching the Rust `DeltaNet` architecture exactly, regressing Δ to
`residual = label - nnue_pred`. Writes the resulting weights as an AZR3
file the Rust side loads via `HybridNetwork::load_with_nnue`.

Architecture (matches `crates/cascadia-ai/src/hybrid.rs`):
    16ch input → HexConv stem 16→32 → ReLU
    → 2× ResHexBlock (HexConv 32→32 → ReLU → HexConv 32→32 → +residual → ReLU)
    → mean-pool over 127 real cells (exclude pad index 127)
    → Linear 32→16 → ReLU
    → Linear 16→1
    → scalar Δ

Usage:
    python3 train_hybrid_delta.py \\
        --hybr /tmp/hybr_smoke.hybr \\
        --out  /tmp/hybr_smoke.azr3 \\
        --epochs 20 --batch-size 128 --lr 3e-4 --alpha 0.3

The output AZR3 stores only alpha + Δ. The companion NNUE binary stays
referenced by path at inference time (two-file format).
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Architecture constants — must match `hybrid.rs`.
# ─────────────────────────────────────────────────────────────────────

DELTA_INPUT_CHANNELS = 61
DELTA_TRUNK_CHANNELS = 64
DELTA_BLOCKS = 3
DELTA_HIDDEN = 32
AZ_LOCAL_CELLS = 127
AZ_CELLS_PADDED = 128
AZ_PAD_INDEX = 127
V6_LOCAL_RADIUS = 6
GRID_DIM = 21


# ─────────────────────────────────────────────────────────────────────
# Hex disk + 6-neighbor topology — mirrors `nnue::v6_peak`.
# ─────────────────────────────────────────────────────────────────────

def _hex_dist_from_center(col: int, row: int) -> int:
    q = col - 10
    r = row - 10
    return max(abs(q), abs(r), abs(q + r))


def build_global_to_local() -> np.ndarray:
    """Spiral lookup: 441 global cells → 127 local indices (-1 if off-disk)."""
    tbl = np.full(441, -1, dtype=np.int16)
    nxt = 0
    for d in range(V6_LOCAL_RADIUS + 1):
        for col in range(GRID_DIM):
            for row in range(GRID_DIM):
                if _hex_dist_from_center(col, row) == d:
                    g = col * GRID_DIM + row
                    tbl[g] = nxt
                    nxt += 1
    assert nxt == AZ_LOCAL_CELLS
    return tbl


def build_local_to_global(g2l: np.ndarray) -> np.ndarray:
    out = np.zeros(AZ_LOCAL_CELLS, dtype=np.uint16)
    for g, l in enumerate(g2l):
        if l >= 0:
            out[l] = g
    return out


# Hex 6-direction offsets, matching cascadia_core::hex::HexCoord::neighbor.
# Each direction is (dcol, drow) in our axial col/row layout.
HEX_DIRS = np.array(
    [
        (+1, 0),
        (+1, -1),
        (0, -1),
        (-1, 0),
        (-1, +1),
        (0, +1),
    ],
    dtype=np.int32,
)


def _axial_to_global(q: int, r: int):
    """Axial (q, r) → flat global index in the 21×21 grid (col-major col*21+row).
    Origin (0,0) at (col=10, row=10). Returns None if out of grid."""
    col, row = q + 10, r + 10
    if 0 <= col < 21 and 0 <= row < 21:
        return col * 21 + row
    return None


def _global_to_axial(g: int):
    col, row = g // 21, g % 21
    return (col - 10, row - 10)


def _rotate60_ccw(q: int, r: int):
    """60° CCW in axial: (q, r) → (-r, q+r). Six applications return identity."""
    return (-r, q + r)


def _reflect_q_axis(q: int, r: int):
    """Reflection across the q-axis (line r = -q/2 in cube space)."""
    return (q + r, -r)


def build_hex_symmetries():
    """Build the 12 hex symmetries (6 rotations × 2 reflections) as a pair of
    permutation tables suitable for batched augmentation:

      cell_perms: int32[12, 128]   — `new_board[i] = old_board[cell_perms[t][i]]`
      dir_perms : int32[12, 6]     — under transform t, original direction
                                     dir_perms[t][new_d] becomes new direction new_d
                                     (i.e. gather index for per-direction channels)

    The 127-cell hex disk is rotation- and reflection-symmetric, so every cell
    has a valid destination under every transform. Pad index 127 maps to 127.
    """
    g2l = build_global_to_local()
    l2g = build_local_to_global(g2l)

    DIRECTION_DELTAS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

    def apply_xform(q, r, n_rot, reflect):
        if reflect:
            q, r = _reflect_q_axis(q, r)
        for _ in range(n_rot):
            q, r = _rotate60_ccw(q, r)
        return q, r

    cell_perms = []
    dir_perms = []
    for reflect in (False, True):
        for n_rot in range(6):
            cell_perm = np.full(AZ_CELLS_PADDED, AZ_PAD_INDEX, dtype=np.int32)
            for old_local in range(AZ_LOCAL_CELLS):
                old_g = int(l2g[old_local])
                old_q, old_r = _global_to_axial(old_g)
                new_q, new_r = apply_xform(old_q, old_r, n_rot, reflect)
                new_g = _axial_to_global(new_q, new_r)
                if new_g is None:
                    continue
                new_local = int(g2l[new_g])
                if new_local >= 0:
                    # Gather index: new_board[new_local] = old_board[old_local]
                    cell_perm[new_local] = old_local
            cell_perm[AZ_PAD_INDEX] = AZ_PAD_INDEX
            cell_perms.append(cell_perm)

            # Direction permutation: under transform t, the original direction
            # that becomes new_d is the one whose delta, transformed, equals d.
            dir_perm = np.zeros(6, dtype=np.int32)
            for new_d in range(6):
                # Apply INVERSE transform to new_d's delta to find which
                # original direction it came from. Inverse of (n_rot, reflect):
                # apply (6 - n_rot) rotations, then reflect (since reflection
                # is self-inverse). But it's easier to walk old directions.
                for old_d in range(6):
                    odq, odr = DIRECTION_DELTAS[old_d]
                    tdq, tdr = apply_xform(odq, odr, n_rot, reflect)
                    if (tdq, tdr) == DIRECTION_DELTAS[new_d]:
                        dir_perm[new_d] = old_d
                        break
            dir_perms.append(dir_perm)

    return np.array(cell_perms, dtype=np.int32), np.array(dir_perms, dtype=np.int32)


def build_neighbors_local() -> np.ndarray:
    """Per local cell, return [self, n0, n1, n2, n3, n4, n5] local indices.
    Out-of-disk neighbors → AZ_PAD_INDEX (127). Pad cell at index 127 has
    all 7 entries = AZ_PAD_INDEX so it stays inert under convolution."""
    g2l = build_global_to_local()
    l2g = build_local_to_global(g2l)
    out = np.full((AZ_CELLS_PADDED, 7), AZ_PAD_INDEX, dtype=np.int32)
    for li in range(AZ_LOCAL_CELLS):
        g = int(l2g[li])
        col = g // GRID_DIM
        row = g % GRID_DIM
        out[li, 0] = li  # self
        for k, (dc, dr) in enumerate(HEX_DIRS):
            nc, nr = col + int(dc), row + int(dr)
            if 0 <= nc < GRID_DIM and 0 <= nr < GRID_DIM:
                ng = nc * GRID_DIM + nr
                nl = int(g2l[ng])
                out[li, 1 + k] = nl if nl >= 0 else AZ_PAD_INDEX
    return out  # [128, 7]


# ─────────────────────────────────────────────────────────────────────
# HYBR file loader.
# ─────────────────────────────────────────────────────────────────────

HYBR_MAGIC = b"HYBR"
HYBR_VERSION = 1


def load_hybr(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (boards [N, 16, 128], nnue_preds [N], labels [N])."""
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"HYBR file {path} too short")
    if raw[:4] != HYBR_MAGIC:
        raise ValueError(f"HYBR magic mismatch in {path}: got {raw[:4]!r}")
    version, channels, cells = struct.unpack_from("<III", raw, 4)
    if version != HYBR_VERSION:
        raise ValueError(f"HYBR version {version}, expected {HYBR_VERSION}")
    if channels != DELTA_INPUT_CHANNELS or cells != AZ_CELLS_PADDED:
        raise ValueError(
            f"HYBR shape {channels}x{cells} != DeltaNet {DELTA_INPUT_CHANNELS}x{AZ_CELLS_PADDED}"
        )
    header_size = 16
    record_size = (channels * cells + 2) * 4
    payload = raw[header_size:]
    if len(payload) % record_size != 0:
        raise ValueError(
            f"HYBR payload {len(payload)} bytes not divisible by record size {record_size}"
        )
    n_records = len(payload) // record_size
    boards = np.zeros((n_records, channels, cells), dtype=np.float32)
    nnue_preds = np.zeros(n_records, dtype=np.float32)
    labels = np.zeros(n_records, dtype=np.float32)
    for i in range(n_records):
        off = i * record_size
        # Board: read as 1D then reshape.
        board_flat = np.frombuffer(payload, dtype=np.float32, count=channels * cells, offset=off)
        boards[i] = board_flat.reshape(channels, cells)
        nnue_preds[i] = np.frombuffer(payload, dtype=np.float32, count=1, offset=off + channels * cells * 4)[0]
        labels[i] = np.frombuffer(payload, dtype=np.float32, count=1, offset=off + (channels * cells + 1) * 4)[0]
    return boards, nnue_preds, labels


# ─────────────────────────────────────────────────────────────────────
# DeltaNet — MLX implementation.
# Layout matches Rust HexConv: weights stored as [out_c, in_c, 7] but
# flattened to row-major [out_c * in_c * 7] on AZR3 export.
# ─────────────────────────────────────────────────────────────────────

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class HexConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        scale = (2.0 / (in_c * 7)) ** 0.5
        # MLX random.normal initializer.
        self.w = mx.random.uniform(low=-scale, high=scale, shape=(out_c, in_c, 7))
        self.b = mx.zeros((out_c,))

    def __call__(self, x: mx.array, neighbors: mx.array) -> mx.array:
        # x: [B, in_c, 128]
        # neighbors: [128, 7]  (int32)
        # gather: gathered[b, c, n, k] = x[b, c, neighbors[n, k]]
        gathered = mx.take(x, neighbors, axis=-1)  # [B, in_c, 128, 7]
        # out[b, o, n] = sum_{c, k} w[o, c, k] * gathered[b, c, n, k]
        out = mx.einsum("bcnk,ock->bon", gathered, self.w)
        out = out + self.b[None, :, None]
        return out


def _dropout(x: "mx.array", p: float, training: bool) -> "mx.array":
    """Inverted dropout: zero each activation with probability p, scale survivors
    by 1/(1-p) so the expected magnitude is unchanged. No-op when not training."""
    if (not training) or p <= 0.0:
        return x
    keep = 1.0 - p
    mask = mx.random.bernoulli(keep, x.shape)
    return x * mask / keep


class ResHexBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.c1 = HexConv(channels, channels)
        self.c2 = HexConv(channels, channels)
        self.dropout = dropout

    def __call__(self, x: mx.array, neighbors: mx.array, training: bool = False) -> mx.array:
        z1 = nn.relu(self.c1(x, neighbors))
        z1 = _dropout(z1, self.dropout, training)
        z2 = self.c2(z1, neighbors)
        out = nn.relu(z2 + x)
        out = _dropout(out, self.dropout, training)
        return out


class DeltaNet(nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        trunk_channels: int = DELTA_TRUNK_CHANNELS,
        blocks: int = DELTA_BLOCKS,
        hidden: int = DELTA_HIDDEN,
    ):
        super().__init__()
        # Store arch params on self so export_azr3 can read them back.
        self.input_channels = DELTA_INPUT_CHANNELS
        self.trunk_channels = trunk_channels
        self.blocks_count = blocks
        self.hidden = hidden
        self.stem = HexConv(DELTA_INPUT_CHANNELS, trunk_channels)
        self.blocks = [
            ResHexBlock(trunk_channels, dropout=dropout)
            for _ in range(blocks)
        ]
        self.dropout = dropout
        # Head: pool → hidden → 1. Match Rust [hidden, trunk_c] layout.
        scale_h1 = (2.0 / trunk_channels) ** 0.5
        self.head_w1 = mx.random.uniform(
            low=-scale_h1, high=scale_h1, shape=(hidden, trunk_channels)
        )
        self.head_b1 = mx.zeros((hidden,))
        # Zero init head_w2 so a freshly-built DeltaNet returns Δ=0 — same
        # safety invariant as the Rust DeltaNet::new.
        self.head_w2 = mx.zeros((hidden,))
        self.head_b2 = mx.zeros((1,))

    def __call__(self, x: mx.array, neighbors: mx.array, training: bool = False) -> mx.array:
        # x: [B, channels, 128]
        z = nn.relu(self.stem(x, neighbors))
        z = _dropout(z, self.dropout, training)
        for blk in self.blocks:
            z = blk(z, neighbors, training=training)
        # Mean-pool over real cells (exclude pad at index 127).
        pooled = mx.mean(z[:, :, :AZ_LOCAL_CELLS], axis=-1)
        h = pooled @ self.head_w1.T + self.head_b1[None, :]
        h = nn.relu(h)
        h = _dropout(h, self.dropout, training)
        out = (h @ self.head_w2).reshape(-1, 1) + self.head_b2[None, :]
        return out.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────
# AZR3 export — byte-exact match for `HybridNetwork::save_delta`.
# ─────────────────────────────────────────────────────────────────────

AZR3_MAGIC = b"AZR3"


def _flatten_hex_conv_w(w: mx.array) -> np.ndarray:
    """Convert MLX [out_c, in_c, 7] → flat row-major [out_c * in_c * 7]."""
    arr = np.array(w)
    out_c, in_c, _ = arr.shape
    # Flat layout matches the Rust loader's reading order: outermost is
    # out_c, then in_c, then k. Reshape (out_c, in_c*7) preserves that.
    return arr.reshape(out_c * in_c * 7).astype(np.float32)


def export_azr3(model: DeltaNet, alpha: float, out_path: Path) -> None:
    with out_path.open("wb") as f:
        f.write(AZR3_MAGIC)
        f.write(struct.pack("<f", alpha))
        f.write(struct.pack("<I", 1))  # delta-format version
        # DeltaNet header — read actual model dims so we can save bigger
        # variants. Rust DeltaNet::read uses these to allocate properly.
        f.write(struct.pack("<IIII",
                            getattr(model, "input_channels", DELTA_INPUT_CHANNELS),
                            getattr(model, "trunk_channels", DELTA_TRUNK_CHANNELS),
                            getattr(model, "blocks_count", DELTA_BLOCKS),
                            getattr(model, "hidden", DELTA_HIDDEN)))
        # Stem
        f.write(struct.pack("<II", model.stem.in_c, model.stem.out_c))
        f.write(_flatten_hex_conv_w(model.stem.w).tobytes())
        f.write(np.array(model.stem.b, dtype=np.float32).tobytes())
        # Blocks
        for blk in model.blocks:
            for hc in (blk.c1, blk.c2):
                f.write(struct.pack("<II", hc.in_c, hc.out_c))
                f.write(_flatten_hex_conv_w(hc.w).tobytes())
                f.write(np.array(hc.b, dtype=np.float32).tobytes())
        # Head
        # head_w1 stored as flat [hidden * trunk_c] — matches Rust row-major.
        f.write(np.array(model.head_w1, dtype=np.float32).reshape(-1).tobytes())
        f.write(np.array(model.head_b1, dtype=np.float32).tobytes())
        f.write(np.array(model.head_w2, dtype=np.float32).tobytes())
        f.write(struct.pack("<f", float(np.array(model.head_b2)[0])))


# ─────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hybr", type=Path, required=True, help="Path to HYBR file")
    p.add_argument("--out", type=Path, required=True, help="Output AZR3 path")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.3,
                   help="α stored in AZR3 (the blend coefficient used at inference)")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    args = p.parse_args()

    print(f"Loading {args.hybr}", file=sys.stderr)
    boards, nnue_preds, labels = load_hybr(args.hybr)
    residuals = labels - nnue_preds
    n = len(boards)
    print(f"  {n} records  "
          f"residual mean={residuals.mean():.3f} std={residuals.std():.3f}  "
          f"label mean={labels.mean():.3f}  nnue_pred mean={nnue_preds.mean():.3f}",
          file=sys.stderr)
    if n < 32:
        print(f"WARN: very small dataset ({n} records); training may not be meaningful",
              file=sys.stderr)

    # Train/val split — random shuffle.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    boards_mx = mx.array(boards)
    residuals_mx = mx.array(residuals)
    train_idx_mx = mx.array(train_idx.astype(np.int32))
    val_idx_mx = mx.array(val_idx.astype(np.int32))

    neighbors = mx.array(build_neighbors_local().astype(np.int32))

    mx.random.seed(args.seed)
    model = DeltaNet()
    optimizer = optim.Adam(learning_rate=args.lr)

    def loss_fn(model, xb, target):
        pred = model(xb, neighbors)
        return mx.mean((pred - target) ** 2)

    loss_and_grad = mx.compile(
        nn.value_and_grad(model, loss_fn),
    ) if False else nn.value_and_grad(model, loss_fn)

    n_train = len(train_idx)
    n_batches = max(1, n_train // args.batch_size)
    print(f"  train={n_train}  val={n_val}  batches/epoch={n_batches}", file=sys.stderr)

    for epoch in range(args.epochs):
        t0 = time.time()
        # Shuffle train indices each epoch.
        epoch_perm = mx.random.permutation(n_train)
        train_shuffled = mx.take(train_idx_mx, epoch_perm, axis=0)
        epoch_loss = 0.0
        for b in range(n_batches):
            batch_idx = train_shuffled[b * args.batch_size:(b + 1) * args.batch_size]
            xb = mx.take(boards_mx, batch_idx, axis=0)
            tb = mx.take(residuals_mx, batch_idx, axis=0)
            loss, grads = loss_and_grad(model, xb, tb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += float(loss)
        epoch_loss /= n_batches

        # Validation.
        xv = mx.take(boards_mx, val_idx_mx, axis=0)
        tv = mx.take(residuals_mx, val_idx_mx, axis=0)
        pred = model(xv, neighbors)
        val_mse = float(mx.mean((pred - tv) ** 2))
        # End-to-end RMSE on (nnue_pred + α * Δ) vs label.
        pred_np = np.array(pred)
        nnue_val = nnue_preds[val_idx]
        labels_val = labels[val_idx]
        e2e_rmse_nnue = float(np.sqrt(np.mean((nnue_val - labels_val) ** 2)))
        e2e_rmse_hybrid = float(
            np.sqrt(np.mean((nnue_val + args.alpha * pred_np - labels_val) ** 2))
        )
        elapsed = time.time() - t0
        print(
            f"epoch {epoch+1:3d}: train_mse={epoch_loss:.4f}  val_mse={val_mse:.4f}  "
            f"e2e_rmse nnue={e2e_rmse_nnue:.3f} hybrid={e2e_rmse_hybrid:.3f}  "
            f"({elapsed:.1f}s)",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_azr3(model, args.alpha, args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
