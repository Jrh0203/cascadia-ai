#!/usr/bin/env python3
"""MLX trainer for the Cascadia AlphaZero policy/value network.

Rust owns exact game rules, legal candidate generation, and PUCT collection.
This script owns neural training on Apple Silicon via MLX and writes weights
back to the AZR1 format consumed by `--az`.
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI users.
    raise SystemExit(
        "MLX is not installed for this Python. Install with:\n"
        "  python3 -m pip install mlx\n"
        "or use the Codex bundled Python shown in docs/ALPHAZERO_MLX.md"
    ) from exc


AZD_MAGIC = b"AZD1"
AZR_MAGIC = b"AZR1"
VALUE_SCALE = 120.0


@dataclass
class AzDataset:
    inputs: np.ndarray
    tile_idx: np.ndarray
    wildlife_idx: np.ndarray
    market_idx: np.ndarray
    wildlife_market_idx: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    channels: int
    grid_dim: int
    grid_size: int

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


def load_azd(paths: list[Path], max_candidates: int | None = None) -> AzDataset:
    raw_samples = []
    channels = grid_dim = grid_size = None
    observed_max = 0

    for path in paths:
        data = path.read_bytes()
        pos = 0
        if data[:4] != AZD_MAGIC:
            raise ValueError(f"{path}: bad magic {data[:4]!r}, expected {AZD_MAGIC!r}")
        pos = 4
        ch, pos = _read_u32(data, pos)
        gd, pos = _read_u32(data, pos)
        gs, pos = _read_u32(data, pos)
        n_samples, pos = _read_u32(data, pos)
        if channels is None:
            channels, grid_dim, grid_size = ch, gd, gs
        elif (channels, grid_dim, grid_size) != (ch, gd, gs):
            raise ValueError(f"{path}: header mismatch")
        input_len = ch * gs
        for _ in range(n_samples):
            value, pos = _read_f32(data, pos)
            n, pos = _read_u32(data, pos)
            observed_max = max(observed_max, n)
            inp = np.frombuffer(data, dtype="<f4", count=input_len, offset=pos).copy()
            pos += input_len * 4
            cands = np.frombuffer(data, dtype="<i4", count=n * 4, offset=pos).copy().reshape(n, 4)
            pos += n * 4 * 4
            policy = np.frombuffer(data, dtype="<f4", count=n, offset=pos).copy()
            pos += n * 4
            raw_samples.append((value, inp, cands, policy))
        if pos != len(data):
            raise ValueError(f"{path}: trailing bytes ({len(data) - pos})")

    if not raw_samples:
        raise ValueError("no AlphaZero samples loaded")
    k = max_candidates or observed_max
    if k < observed_max:
        raise ValueError(f"--max-candidates {k} is smaller than observed candidate count {observed_max}")

    n = len(raw_samples)
    inputs = np.zeros((n, channels, grid_dim, grid_dim), dtype=np.float32)
    tile_idx = np.zeros((n, k), dtype=np.int32)
    wildlife_idx = np.full((n, k), -1, dtype=np.int32)
    market_idx = np.zeros((n, k), dtype=np.int32)
    wildlife_market_idx = np.zeros((n, k), dtype=np.int32)
    mask = np.zeros((n, k), dtype=bool)
    policy = np.zeros((n, k), dtype=np.float32)
    values = np.zeros((n,), dtype=np.float32)

    for i, (value, inp, cands, pol) in enumerate(raw_samples):
        m = cands.shape[0]
        inputs[i] = inp.reshape(channels, grid_size).reshape(channels, grid_dim, grid_dim)
        tile_idx[i, :m] = np.maximum(cands[:, 0], 0)
        wildlife_idx[i, :m] = cands[:, 1]
        market_idx[i, :m] = np.clip(cands[:, 2], 0, 3)
        wildlife_market_idx[i, :m] = np.where(cands[:, 3] >= 0, cands[:, 3], cands[:, 2])
        wildlife_market_idx[i, :m] = np.clip(wildlife_market_idx[i, :m], 0, 3)
        mask[i, :m] = True
        psum = float(pol.sum())
        policy[i, :m] = pol / psum if psum > 0 else 1.0 / m
        values[i] = np.clip(value, 0.0, 1.0)

    return AzDataset(
        inputs=inputs,
        tile_idx=tile_idx,
        wildlife_idx=wildlife_idx,
        market_idx=market_idx,
        wildlife_market_idx=wildlife_market_idx,
        mask=mask,
        policy=policy,
        value=values,
        channels=channels,
        grid_dim=grid_dim,
        grid_size=grid_size,
    )


def rand_uniform(shape, scale: float):
    return mx.random.uniform(low=-scale, high=scale, shape=shape)


class AzConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        scale = (2.0 / max(1, in_c * 9)) ** 0.5
        self.in_c = in_c
        self.out_c = out_c
        self.w = rand_uniform((out_c, in_c, 3, 3), scale)
        self.b = mx.zeros((out_c,), dtype=mx.float32)

    def __call__(self, x):
        bsz, _, h, wdim = x.shape
        xpad = mx.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1)])
        out = mx.broadcast_to(self.b.reshape(1, self.out_c, 1, 1), (bsz, self.out_c, h, wdim))
        for ky in range(3):
            for kx in range(3):
                patch = xpad[:, :, ky : ky + h, kx : kx + wdim]
                out = out + mx.einsum("bchw,oc->bohw", patch, self.w[:, :, ky, kx])
        return out


class AzResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = AzConv2d(channels, channels)
        self.c2 = AzConv2d(channels, channels)

    def __call__(self, x):
        return mx.maximum(self.c2(mx.maximum(self.c1(x), 0.0)) + x, 0.0)


class CascadiaAzNet(nn.Module):
    def __init__(self, input_channels: int, channels: int, blocks: int, hidden: int, max_candidates: int, c_puct: float):
        super().__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.blocks_count = blocks
        self.hidden = hidden
        self.max_candidates = max_candidates
        self.c_puct = c_puct
        self.stem = AzConv2d(input_channels, channels)
        self.blocks = [AzResidualBlock(channels) for _ in range(blocks)]
        head_scale = (2.0 / max(1, channels)) ** 0.5
        self.policy_tile_w = rand_uniform((channels,), head_scale)
        self.policy_tile_b = mx.array(0.0, dtype=mx.float32)
        self.policy_wildlife_w = rand_uniform((channels,), head_scale)
        self.policy_wildlife_b = mx.array(0.0, dtype=mx.float32)
        self.policy_market_w = rand_uniform((4, channels), head_scale)
        self.policy_market_b = mx.zeros((4,), dtype=mx.float32)
        self.policy_wildlife_market_w = rand_uniform((4, channels), head_scale)
        self.policy_wildlife_market_b = mx.zeros((4,), dtype=mx.float32)
        self.policy_skip_w = rand_uniform((channels,), head_scale)
        self.policy_skip_b = mx.array(0.0, dtype=mx.float32)
        self.value_w1 = rand_uniform((hidden, channels), head_scale)
        self.value_b1 = mx.zeros((hidden,), dtype=mx.float32)
        self.value_w2 = rand_uniform((hidden,), (2.0 / max(1, hidden)) ** 0.5)
        self.value_b2 = mx.array(0.0, dtype=mx.float32)

    def trunk(self, x):
        x = mx.maximum(self.stem(x), 0.0)
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, batch):
        x = self.trunk(batch["inputs"])
        pooled = mx.mean(x, axis=(2, 3))
        tile_logits = mx.einsum("bchw,c->bhw", x, self.policy_tile_w).reshape(x.shape[0], -1) + self.policy_tile_b
        wildlife_logits = (
            mx.einsum("bchw,c->bhw", x, self.policy_wildlife_w).reshape(x.shape[0], -1)
            + self.policy_wildlife_b
        )
        market_logits = pooled @ mx.transpose(self.policy_market_w) + self.policy_market_b
        wildlife_market_logits = (
            pooled @ mx.transpose(self.policy_wildlife_market_w) + self.policy_wildlife_market_b
        )
        skip_logits = pooled @ self.policy_skip_w + self.policy_skip_b
        vh = mx.maximum(pooled @ mx.transpose(self.value_w1) + self.value_b1, 0.0)
        value = mx.sigmoid(vh @ self.value_w2 + self.value_b2)

        tile = mx.take_along_axis(tile_logits, batch["tile_idx"], axis=1)
        safe_wildlife_idx = mx.maximum(batch["wildlife_idx"], 0)
        wildlife = mx.where(
            batch["wildlife_idx"] >= 0,
            mx.take_along_axis(wildlife_logits, safe_wildlife_idx, axis=1),
            skip_logits[:, None],
        )
        market = mx.take_along_axis(market_logits, batch["market_idx"], axis=1)
        wildlife_market = mx.take_along_axis(wildlife_market_logits, batch["wildlife_market_idx"], axis=1)
        logits = tile + wildlife + market + wildlife_market
        return logits, value


def make_batch(ds: AzDataset, idx: np.ndarray) -> dict[str, mx.array]:
    return {
        "inputs": mx.array(ds.inputs[idx]),
        "tile_idx": mx.array(ds.tile_idx[idx]),
        "wildlife_idx": mx.array(ds.wildlife_idx[idx]),
        "market_idx": mx.array(ds.market_idx[idx]),
        "wildlife_market_idx": mx.array(ds.wildlife_market_idx[idx]),
        "mask": mx.array(ds.mask[idx]),
        "policy": mx.array(ds.policy[idx]),
        "value": mx.array(ds.value[idx]),
    }


def loss_fn(model: CascadiaAzNet, batch: dict[str, mx.array], value_weight: float):
    logits, value = model(batch)
    masked = mx.where(batch["mask"], logits, mx.array(-1.0e9, dtype=mx.float32))
    log_probs = masked - mx.logsumexp(masked, axis=1, keepdims=True)
    policy_loss = -mx.sum(batch["policy"] * log_probs, axis=1).mean()
    value_loss = mx.mean(mx.square(value - batch["value"]))
    return policy_loss + value_weight * value_loss


def eval_losses(model: CascadiaAzNet, ds: AzDataset, batch_size: int, value_weight: float):
    total = 0.0
    top1 = 0
    seen = 0
    for start in range(0, ds.size, batch_size):
        idx = np.arange(start, min(ds.size, start + batch_size))
        batch = make_batch(ds, idx)
        loss = loss_fn(model, batch, value_weight)
        logits, _value = model(batch)
        masked = mx.where(batch["mask"], logits, mx.array(-1.0e9, dtype=mx.float32))
        pred = np.array(mx.argmax(masked, axis=1))
        target = np.array(np.argmax(ds.policy[idx], axis=1))
        top1 += int((pred == target).sum())
        seen += len(idx)
        total += float(np.array(loss)) * len(idx)
    return total / max(1, seen), top1 / max(1, seen)


def train(model: CascadiaAzNet, train_ds: AzDataset, val_ds: AzDataset | None, args):
    opt = optim.Adam(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, lambda m, b: loss_fn(m, b, args.value_weight))
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(train_ds.size)
        running = 0.0
        seen = 0
        for start in range(0, train_ds.size, args.batch_size):
            idx = order[start : start + args.batch_size]
            batch = make_batch(train_ds, idx)
            loss, grads = loss_and_grad(model, batch)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            running += float(np.array(loss)) * len(idx)
            seen += len(idx)
        train_loss = running / max(1, seen)
        msg = f"epoch {epoch:03d}: train_loss={train_loss:.5f} elapsed={time.time() - t0:.1f}s"
        if val_ds is not None and val_ds.size > 0:
            val_loss, top1 = eval_losses(model, val_ds, args.batch_size, args.value_weight)
            msg += f" val_loss={val_loss:.5f} val_top1={top1:.3f}"
        print(msg, flush=True)
        if args.save_each_epoch:
            save_azr(model, Path(f"{Path(args.out).with_suffix('').as_posix()}_epoch{epoch}.azr"))
    return model


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


def save_conv(f, conv: AzConv2d):
    _write_u32(f, conv.in_c)
    _write_u32(f, conv.out_c)
    _write_vec(f, _as_np(conv.w))
    _write_vec(f, _as_np(conv.b))


def save_azr(model: CascadiaAzNet, path: Path):
    mx.eval(model.parameters())
    with path.open("wb") as f:
        f.write(AZR_MAGIC)
        _write_u32(f, model.channels)
        _write_u32(f, model.blocks_count)
        _write_u32(f, model.hidden)
        _write_u32(f, model.max_candidates)
        _write_f32(f, model.c_puct)
        save_conv(f, model.stem)
        for block in model.blocks:
            save_conv(f, block.c1)
            save_conv(f, block.c2)
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
        _write_vec(f, _as_np(model.value_w1))
        _write_vec(f, _as_np(model.value_b1))
        _write_vec(f, _as_np(model.value_w2))
        _write_f32(f, float(np.array(model.value_b2)))
    print(f"saved {path}", flush=True)


def _read_vec(data: bytes, pos: int) -> tuple[np.ndarray, int]:
    n, pos = _read_u32(data, pos)
    arr = np.frombuffer(data, dtype="<f4", count=n, offset=pos).copy()
    return arr, pos + n * 4


def _load_conv(data: bytes, pos: int, conv: AzConv2d) -> int:
    in_c, pos = _read_u32(data, pos)
    out_c, pos = _read_u32(data, pos)
    if (in_c, out_c) != (conv.in_c, conv.out_c):
        raise ValueError(f"conv shape mismatch: file {(in_c, out_c)} model {(conv.in_c, conv.out_c)}")
    w, pos = _read_vec(data, pos)
    b, pos = _read_vec(data, pos)
    conv.w = mx.array(w.reshape(out_c, in_c, 3, 3))
    conv.b = mx.array(b.reshape(out_c))
    return pos


def load_azr(path: Path, input_channels: int) -> CascadiaAzNet:
    data = path.read_bytes()
    pos = 0
    if data[:4] != AZR_MAGIC:
        raise ValueError(f"{path}: bad magic {data[:4]!r}")
    pos = 4
    channels, pos = _read_u32(data, pos)
    blocks, pos = _read_u32(data, pos)
    hidden, pos = _read_u32(data, pos)
    max_candidates, pos = _read_u32(data, pos)
    c_puct, pos = _read_f32(data, pos)
    model = CascadiaAzNet(input_channels, channels, blocks, hidden, max_candidates, c_puct)
    pos = _load_conv(data, pos, model.stem)
    for block in model.blocks:
        pos = _load_conv(data, pos, block.c1)
        pos = _load_conv(data, pos, block.c2)
    v, pos = _read_vec(data, pos); model.policy_tile_w = mx.array(v.reshape(channels))
    model.policy_tile_b = mx.array(_read_f32(data, pos)[0]); pos += 4
    v, pos = _read_vec(data, pos); model.policy_wildlife_w = mx.array(v.reshape(channels))
    model.policy_wildlife_b = mx.array(_read_f32(data, pos)[0]); pos += 4
    v, pos = _read_vec(data, pos); model.policy_market_w = mx.array(v.reshape(4, channels))
    b = []
    for _ in range(4):
        x, pos = _read_f32(data, pos); b.append(x)
    model.policy_market_b = mx.array(np.array(b, dtype=np.float32))
    v, pos = _read_vec(data, pos); model.policy_wildlife_market_w = mx.array(v.reshape(4, channels))
    b = []
    for _ in range(4):
        x, pos = _read_f32(data, pos); b.append(x)
    model.policy_wildlife_market_b = mx.array(np.array(b, dtype=np.float32))
    v, pos = _read_vec(data, pos); model.policy_skip_w = mx.array(v.reshape(channels))
    model.policy_skip_b = mx.array(_read_f32(data, pos)[0]); pos += 4
    v, pos = _read_vec(data, pos); model.value_w1 = mx.array(v.reshape(hidden, channels))
    v, pos = _read_vec(data, pos); model.value_b1 = mx.array(v.reshape(hidden))
    v, pos = _read_vec(data, pos); model.value_w2 = mx.array(v.reshape(hidden))
    model.value_b2 = mx.array(_read_f32(data, pos)[0]); pos += 4
    if pos != len(data):
        raise ValueError(f"{path}: trailing bytes {len(data) - pos}")
    return model


def split_dataset(ds: AzDataset, val_fraction: float, seed: int):
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
        return AzDataset(
            inputs=ds.inputs[idx],
            tile_idx=ds.tile_idx[idx],
            wildlife_idx=ds.wildlife_idx[idx],
            market_idx=ds.market_idx[idx],
            wildlife_market_idx=ds.wildlife_market_idx[idx],
            mask=ds.mask[idx],
            policy=ds.policy[idx],
            value=ds.value[idx],
            channels=ds.channels,
            grid_dim=ds.grid_dim,
            grid_size=ds.grid_size,
        )

    return take(train_idx), take(val_idx)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", nargs="+", required=True, help="One or more AZD1 sample files")
    p.add_argument("--out", required=True, help="Output AZR1 weights")
    p.add_argument("--init", help="Optional AZR1 checkpoint to resume from")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--max-candidates", type=int)
    p.add_argument("--c-puct", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-each-epoch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    mx.random.seed(args.seed)
    paths = [Path(p) for p in args.samples]
    ds = load_azd(paths, args.max_candidates)
    print(
        f"MLX device={mx.default_device()} samples={ds.size} channels={ds.channels} "
        f"grid={ds.grid_dim} max_candidates={ds.max_candidates}",
        flush=True,
    )
    train_ds, val_ds = split_dataset(ds, args.val_fraction, args.seed)
    if args.init:
        model = load_azr(Path(args.init), ds.channels)
        if model.max_candidates != ds.max_candidates:
            print(
                f"warning: init max_candidates={model.max_candidates}, data max_candidates={ds.max_candidates}; "
                "updating checkpoint header to match the replay set",
                flush=True,
            )
            model.max_candidates = ds.max_candidates
    else:
        model = CascadiaAzNet(
            ds.channels,
            args.channels,
            args.blocks,
            args.hidden,
            ds.max_candidates,
            args.c_puct,
        )
    train(model, train_ds, val_ds, args)
    save_azr(model, Path(args.out))


if __name__ == "__main__":
    main()
