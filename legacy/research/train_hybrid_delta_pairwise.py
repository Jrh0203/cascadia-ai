#!/usr/bin/env python3
"""Pairwise-discrimination training for HybridNet's Δ head.

Reads a HYBP file (per-decision: K candidates, each with board / nnue_pred /
mce_value). Trains Δ on pairwise residual differences:

    target_AB = (mce_value_A - nnue_pred_A) - (mce_value_B - nnue_pred_B)
    pred_AB   = delta(board_A) - delta(board_B)
    loss      = (target_AB - pred_AB)^2

The constant residual bias cancels in pairwise differences, forcing Δ to
learn what *discriminates between sibling candidates within the same
decision* — exactly what argmax move selection needs.

Output: an AZR3 file that Rust loads via `HybridNetwork::load_with_nnue`,
byte-compatible with the previous absolute-residual trainer.
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Shared arch + helpers (identical to the absolute-residual trainer).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hybrid_delta import (  # noqa: E402
    DELTA_INPUT_CHANNELS,
    DELTA_TRUNK_CHANNELS,
    DELTA_BLOCKS,
    DELTA_HIDDEN,
    AZ_LOCAL_CELLS,
    AZ_CELLS_PADDED,
    DeltaNet,
    build_neighbors_local,
    build_hex_symmetries,
    export_azr3,
)

# Channel layout for direction permutation (see hybrid.rs DELTA_PLANE_*).
# Per-direction wildlife adjacency lives at channels 16..46 = 16 + d*5 + w.
DIR_BLOCK_START = 16
DIR_BLOCK_PER_DIR = 5  # 5 wildlife types
DIR_BLOCK_END = DIR_BLOCK_START + 6 * DIR_BLOCK_PER_DIR  # 46


def build_channel_perm_for_transform(dir_perm: np.ndarray) -> np.ndarray:
    """For a 61-channel layout, return a channel-permutation int32[61] such that
       `new_channels[c] = old_channels[chan_perm[c]]`. Most channels are
       identity; only the 16..46 directional block gets permuted by dir_perm."""
    chan = np.arange(DELTA_INPUT_CHANNELS, dtype=np.int32)
    for new_d in range(6):
        for w in range(DIR_BLOCK_PER_DIR):
            new_idx = DIR_BLOCK_START + new_d * DIR_BLOCK_PER_DIR + w
            old_d = int(dir_perm[new_d])
            old_idx = DIR_BLOCK_START + old_d * DIR_BLOCK_PER_DIR + w
            chan[new_idx] = old_idx
    return chan

# ─────────────────────────────────────────────────────────────────────
# HYBP loader.
# ─────────────────────────────────────────────────────────────────────

HYBP_MAGIC = b"HYBP"
HYBP_VERSION = 1


def load_hybp(path: Path):
    """Returns list of per-decision arrays:
      (boards_d [K, C, N], nnue_preds_d [K], mce_values_d [K])
    """
    raw = path.read_bytes()
    if raw[:4] != HYBP_MAGIC:
        raise ValueError(f"HYBP magic mismatch: got {raw[:4]!r}")
    version, channels, cells = struct.unpack_from("<III", raw, 4)
    if version != HYBP_VERSION:
        raise ValueError(f"HYBP version {version}, expected {HYBP_VERSION}")
    if channels != DELTA_INPUT_CHANNELS or cells != AZ_CELLS_PADDED:
        raise ValueError(f"shape mismatch {channels}x{cells}")
    pos = 16
    record_size = channels * cells * 4 + 8  # board floats + nnue_pred + mce_value
    decisions = []
    while pos < len(raw):
        k = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        boards = np.zeros((k, channels, cells), dtype=np.float32)
        nnue_preds = np.zeros(k, dtype=np.float32)
        mce_values = np.zeros(k, dtype=np.float32)
        for c in range(k):
            board_flat = np.frombuffer(raw, dtype=np.float32, count=channels * cells, offset=pos)
            boards[c] = board_flat.reshape(channels, cells)
            pos += channels * cells * 4
            nnue_preds[c] = np.frombuffer(raw, dtype=np.float32, count=1, offset=pos)[0]
            pos += 4
            mce_values[c] = np.frombuffer(raw, dtype=np.float32, count=1, offset=pos)[0]
            pos += 4
        decisions.append((boards, nnue_preds, mce_values))
    return decisions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hybp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--pairs-per-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-decision-fraction", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--loss", choices=["mse", "margin"], default="mse",
                   help="mse: regress on residual diff; margin: ranking loss on "
                        "pairs where mce_a > mce_b — robust to MCE value noise")
    p.add_argument("--margin", type=float, default=1.0,
                   help="margin width for ranking loss (in NNUE-score units)")
    p.add_argument("--save-each-epoch", action="store_true",
                   help="Save AZR3 after every epoch as <out>.epoch{N}.azr3, so "
                        "we can pick the best-val checkpoint after the fact")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight decay (default 0 = plain Adam). Try 1e-3.")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout p applied after each ResHexBlock ReLU and head ReLU.")
    p.add_argument("--hex-aug", action="store_true",
                   help="Apply random hex-symmetry augmentation (12 transforms) "
                        "per training batch. ~12x effective data, free.")
    # Distillation scale-up: configurable architecture.
    p.add_argument("--trunk-channels", type=int, default=64,
                   help="DeltaNet trunk channel count. v2 default 64; "
                        "distillation typically 128-256.")
    p.add_argument("--blocks", type=int, default=3,
                   help="ResHexBlock count. v2 default 3; distillation 4-8.")
    p.add_argument("--hidden", type=int, default=32,
                   help="Head hidden size. v2 default 32; distillation 64-128.")
    # Multi-task loss weighting.
    p.add_argument("--value-loss-weight", type=float, default=0.0,
                   help="If > 0, add MSE on per-candidate value prediction "
                        "alongside the pairwise/MSE loss. Multi-task helps "
                        "calibration without hurting ranking.")
    args = p.parse_args()

    print(f"Loading {args.hybp}", file=sys.stderr)
    decisions = load_hybp(args.hybp)
    n_dec = len(decisions)
    n_cand = sum(len(d[1]) for d in decisions)
    print(
        f"  {n_dec} decisions, {n_cand} candidates "
        f"({n_cand / n_dec:.1f} avg/decision)",
        file=sys.stderr,
    )

    # Build flat arrays so we can pull pair samples cheaply via index arithmetic.
    # `decision_index[i]` = decision id for candidate i; pairs only sampled
    # within the same decision id.
    all_boards = np.concatenate([d[0] for d in decisions], axis=0)  # [N, C, P]
    all_nnue = np.concatenate([d[1] for d in decisions], axis=0)    # [N]
    all_mce = np.concatenate([d[2] for d in decisions], axis=0)     # [N]
    decision_id = np.concatenate(
        [np.full(len(d[1]), i, dtype=np.int32) for i, d in enumerate(decisions)]
    )

    # Train/val split BY DECISION (so a decision doesn't leak across splits).
    rng = np.random.default_rng(args.seed)
    decision_perm = rng.permutation(n_dec)
    n_val_dec = max(1, int(n_dec * args.val_decision_fraction))
    val_decisions = set(decision_perm[:n_val_dec].tolist())
    train_decisions = set(decision_perm[n_val_dec:].tolist())

    print(
        f"  train decisions: {len(train_decisions)}  val decisions: {len(val_decisions)}",
        file=sys.stderr,
    )

    # Pre-build pair index per decision for fast batch sampling.
    train_pairs_by_dec = []
    val_pairs_by_dec = []
    cursor = 0
    for di, dec in enumerate(decisions):
        k = len(dec[1])
        # All distinct pairs (a, b) with a != b.
        pairs = [
            (cursor + a, cursor + b)
            for a in range(k)
            for b in range(k)
            if a != b
        ]
        if di in train_decisions:
            train_pairs_by_dec.append(pairs)
        else:
            val_pairs_by_dec.append(pairs)
        cursor += k
    train_pairs = [p for d in train_pairs_by_dec for p in d]
    val_pairs = [p for d in val_pairs_by_dec for p in d]
    print(
        f"  train pairs: {len(train_pairs)}  val pairs: {len(val_pairs)}",
        file=sys.stderr,
    )
    if len(train_pairs) < args.pairs_per_batch:
        print("WARN: too few training pairs", file=sys.stderr)

    boards_mx = mx.array(all_boards)
    nnue_mx = mx.array(all_nnue)
    mce_mx = mx.array(all_mce)
    neighbors = mx.array(build_neighbors_local().astype(np.int32))

    mx.random.seed(args.seed)
    model = DeltaNet(
        dropout=args.dropout,
        trunk_channels=args.trunk_channels,
        blocks=args.blocks,
        hidden=args.hidden,
    )
    # Count params (informational).
    def _count_params(m):
        total = 0
        # Iterate leaf parameters. MLX exposes via parameters().
        for v in mx.tree_flatten(m.parameters())[0]:
            try:
                total += v.size
            except AttributeError:
                pass
        return total
    try:
        n_params = _count_params(model)
        print(f"  arch: trunk_channels={args.trunk_channels} blocks={args.blocks} "
              f"hidden={args.hidden}  params={n_params}",
              file=sys.stderr)
    except Exception:
        pass
    if args.weight_decay > 0:
        optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(learning_rate=args.lr)

    # Hex-symmetry augmentation tables. Built once; sampled per batch.
    if args.hex_aug:
        cell_perms_np, dir_perms_np = build_hex_symmetries()
        cell_perms_mx = mx.array(cell_perms_np)  # [12, 128]
        chan_perms_np = np.stack(
            [build_channel_perm_for_transform(dir_perms_np[t]) for t in range(12)],
            axis=0,
        ).astype(np.int32)
        chan_perms_mx = mx.array(chan_perms_np)  # [12, 61]
        print(f"  hex-aug: 12 symmetries enabled", file=sys.stderr)
    else:
        cell_perms_mx = None
        chan_perms_mx = None

    def apply_symmetry(boards, t_idx):
        """Apply transform `t_idx` to `boards [B, C, N]`. Two gather ops:
        1) permute cells via cell_perms[t_idx] (axis=-1)
        2) permute directional channels via chan_perms[t_idx] (axis=-2)"""
        if cell_perms_mx is None:
            return boards
        boards = mx.take(boards, cell_perms_mx[t_idx], axis=-1)
        boards = mx.take(boards, chan_perms_mx[t_idx], axis=-2)
        return boards

    def loss_fn(model, a_idx, b_idx, t_idx, training):
        a_boards = mx.take(boards_mx, a_idx, axis=0)
        b_boards = mx.take(boards_mx, b_idx, axis=0)
        if cell_perms_mx is not None:
            a_boards = apply_symmetry(a_boards, t_idx)
            b_boards = apply_symmetry(b_boards, t_idx)
        a_nnue = mx.take(nnue_mx, a_idx, axis=0)
        b_nnue = mx.take(nnue_mx, b_idx, axis=0)
        a_mce = mx.take(mce_mx, a_idx, axis=0)
        b_mce = mx.take(mce_mx, b_idx, axis=0)
        d_a = model(a_boards, neighbors, training=training)
        d_b = model(b_boards, neighbors, training=training)
        # Multi-task value-regression loss (distillation paradigm 1).
        # Trains Δ to predict the absolute residual (mce - nnue), not just
        # the pairwise difference. Calibrates Δ's magnitude so α=1 produces
        # mce-quality predictions. value_loss_weight ∈ [0, 1]; typical 0.3.
        value_loss = mx.zeros((1,))
        if args.value_loss_weight > 0:
            target_a = a_mce - a_nnue
            target_b = b_mce - b_nnue
            value_loss = mx.mean((d_a - target_a) ** 2) + mx.mean((d_b - target_b) ** 2)
            value_loss = value_loss * 0.5
        if args.loss == "mse":
            target = (a_mce - a_nnue) - (b_mce - b_nnue)
            pred = d_a - d_b
            rank_loss = mx.mean((pred - target) ** 2)
            return rank_loss + args.value_loss_weight * value_loss
        else:
            # Margin / hinge loss. Drop pairs where mce_a ≈ mce_b (noisy
            # tie zone); for pairs where mce_a clearly > mce_b, require
            # (nnue_a + alpha*delta_a) - (nnue_b + alpha*delta_b) ≥ margin.
            # Robust to MCE estimation noise — we only require the model
            # to get the RANK right, not the magnitude.
            mce_gap = a_mce - b_mce
            # Mask: only train on pairs with non-trivial MCE preference (≥ 1pt).
            valid = mx.abs(mce_gap) > 0.5
            # For valid pairs, sign(mce_gap) is the desired direction.
            direction = mx.sign(mce_gap)
            hybrid_a = a_nnue + args.alpha * d_a
            hybrid_b = b_nnue + args.alpha * d_b
            hybrid_gap = hybrid_a - hybrid_b
            # Margin loss: max(0, margin - direction * hybrid_gap).
            raw = mx.maximum(0.0, args.margin - direction * hybrid_gap)
            # Mean over valid pairs only.
            valid_f = valid.astype(mx.float32)
            margin_loss = mx.sum(raw * valid_f) / mx.maximum(1.0, mx.sum(valid_f))
            return margin_loss + args.value_loss_weight * value_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    train_pairs_np = np.array(train_pairs, dtype=np.int32)
    val_pairs_np = np.array(val_pairs, dtype=np.int32)
    n_train = len(train_pairs_np)
    n_batches = max(1, n_train // args.pairs_per_batch)

    for epoch in range(args.epochs):
        t0 = time.time()
        perm = rng.permutation(n_train)
        total_loss = 0.0
        for b in range(n_batches):
            ids = perm[b * args.pairs_per_batch:(b + 1) * args.pairs_per_batch]
            a_idx = mx.array(train_pairs_np[ids, 0].astype(np.int32))
            b_idx = mx.array(train_pairs_np[ids, 1].astype(np.int32))
            # Pick random hex transform for this batch (identity if hex_aug off).
            t_idx = int(rng.integers(0, 12)) if args.hex_aug else 0
            loss, grads = loss_and_grad(model, a_idx, b_idx, t_idx, True)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += float(loss)
        train_loss = total_loss / n_batches

        # Validation: per-decision pairwise MSE + per-decision argmax agreement
        # rate (how often does Hybrid agree with the MCE-argmax candidate?).
        val_loss_sum = 0.0
        val_n = 0
        n_val_batches = max(1, len(val_pairs_np) // args.pairs_per_batch)
        val_perm = rng.permutation(len(val_pairs_np))
        for b in range(n_val_batches):
            ids = val_perm[b * args.pairs_per_batch:(b + 1) * args.pairs_per_batch]
            a_idx = mx.array(val_pairs_np[ids, 0].astype(np.int32))
            b_idx = mx.array(val_pairs_np[ids, 1].astype(np.int32))
            # Validation: no augmentation, no dropout (training=False).
            l = loss_fn(model, a_idx, b_idx, 0, False)
            val_loss_sum += float(l) * len(ids)
            val_n += len(ids)
        val_loss = val_loss_sum / max(1, val_n)

        # Per-decision argmax agreement.
        nnue_argmax_match = 0
        hybrid_argmax_match = 0
        n_decisions_evaluated = 0
        for di in val_decisions:
            dec_b, dec_n, dec_m = decisions[di]
            if len(dec_n) < 2:
                continue
            xb = mx.array(dec_b)
            deltas = np.array(model(xb, neighbors, training=False))
            mce_best = int(np.argmax(dec_m))
            nnue_best = int(np.argmax(dec_n))
            hybrid_pred = dec_n + args.alpha * deltas
            hybrid_best = int(np.argmax(hybrid_pred))
            if nnue_best == mce_best:
                nnue_argmax_match += 1
            if hybrid_best == mce_best:
                hybrid_argmax_match += 1
            n_decisions_evaluated += 1
        n_eval = max(1, n_decisions_evaluated)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch+1:3d}: pair_mse train={train_loss:.3f} val={val_loss:.3f}  "
            f"argmax-vs-MCE NNUE={nnue_argmax_match/n_eval:.1%} Hybrid={hybrid_argmax_match/n_eval:.1%}  "
            f"({elapsed:.1f}s)",
            file=sys.stderr,
        )
        if args.save_each_epoch:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            ckpt = args.out.with_suffix(f".epoch{epoch+1}.azr3")
            export_azr3(model, args.alpha, ckpt)
            print(f"  ↳ wrote checkpoint {ckpt}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_azr3(model, args.alpha, args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
