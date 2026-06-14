"""Benchmark a trained transformer by actually playing games.

Uses the external-eval protocol in `cascadia-cli --external-eval`. At each
player-0 decision the Rust side streams candidate afterstates as tile-tokens;
this script loads the transformer, scores each candidate's afterstate, and
writes the picked index back.

Usage:
    python3 bench_transformer.py --checkpoint transformer_v1_50k.pt --games 50

    # Baselines for comparison
    python3 bench_transformer.py --checkpoint transformer_v1_50k.pt --games 50 --strategy greedy
    python3 bench_transformer.py --checkpoint transformer_v1_50k.pt --games 50 --strategy random
"""

import argparse
import struct
import subprocess
import sys
import time
import random

import numpy as np
import torch

from train_transformer import (
    CascadiaTransformer,
    DEFAULT_NUM_GLOBAL,
    MAX_TILES,
    TERRAIN_EDGES,
    TILE_METADATA_DIM,
    parse_tile_features,
)
from bench_gnn import (
    MSG_EVAL, MSG_DONE, MSG_FINAL, MSG_PICK,
    read_frame, write_pick, parse_candidate,
)


CASCADIA_CLI = "./target/release/cascadia-cli"


def normalize_globals(raw):
    """Convert 45 raw u8 bytes into 53 normalized f32 globals.

    Same normalization as load_tile_samples in train_transformer.py.
    """
    gf = np.zeros(DEFAULT_NUM_GLOBAL, dtype=np.float32)
    gi = 0
    gf[gi] = raw[0] / 20.0; gi += 1
    gf[gi] = raw[1] / 8.0;  gi += 1
    for i in range(5): gf[gi] = raw[2 + i] / 10.0;  gi += 1
    for i in range(5): gf[gi] = raw[7 + i] / 13.0;  gi += 1
    for i in range(5): gf[gi] = raw[12 + i] / 20.0; gi += 1
    for i in range(5): gf[gi] = raw[17 + i] / 13.0; gi += 1
    for i in range(4): gf[gi] = raw[22 + i] / 5.0;  gi += 1
    for i in range(4): gf[gi] = raw[26 + i] / 5.0;  gi += 1
    for i in range(4): gf[gi] = raw[30 + i] / 5.0;  gi += 1
    for i in range(5): gf[gi] = raw[34 + i] / 29.0; gi += 1
    for i in range(5): gf[gi] = raw[39 + i] / 29.0; gi += 1
    gf[gi] = float(raw[44]); gi += 1
    return gf


def candidate_to_tile_raw(candidate):
    """Rebuild the 11-byte-per-tile numpy array the transformer collate expects."""
    num_tiles = candidate["num_tiles"]
    tile_raw = np.zeros((num_tiles, 11), dtype=np.uint8)
    for i, (terrain, wildlife, allowed, flags, q, r) in enumerate(candidate["tiles"]):
        tile_raw[i, :6] = terrain
        tile_raw[i, 6] = wildlife
        tile_raw[i, 7] = allowed
        tile_raw[i, 8] = flags
        tile_raw[i, 9] = q if q >= 0 else 256 + q  # reinterpret i8 → u8
        tile_raw[i, 10] = r if r >= 0 else 256 + r
    return tile_raw


def pick_transformer_batch(model, device, candidates):
    """Score all candidates in one forward pass. Returns argmax of (current + delta)."""
    B = len(candidates)
    max_tiles = max(c["num_tiles"] for c in candidates) if candidates else 0
    seq_len = max_tiles + 1  # +1 for CLS

    terrain_padded = torch.zeros(B, seq_len, TERRAIN_EDGES, dtype=torch.long)
    wildlife_padded = torch.zeros(B, seq_len, dtype=torch.long)
    metadata_padded = torch.zeros(B, seq_len, TILE_METADATA_DIM, dtype=torch.float32)
    attn_mask = torch.ones(B, seq_len, dtype=torch.bool)  # True = padding
    globals_batch = np.zeros((B, DEFAULT_NUM_GLOBAL), dtype=np.float32)

    for i, c in enumerate(candidates):
        tile_raw = candidate_to_tile_raw(c)
        terrain, wildlife, metadata = parse_tile_features(tile_raw)
        n = c["num_tiles"]
        # CLS at position 0, tiles at 1..n
        terrain_padded[i, 1:n + 1] = terrain
        wildlife_padded[i, 1:n + 1] = wildlife
        metadata_padded[i, 1:n + 1] = metadata
        attn_mask[i, :n + 1] = False
        globals_batch[i] = normalize_globals(c["global_bytes"])

    terrain_padded = terrain_padded.to(device)
    wildlife_padded = wildlife_padded.to(device)
    metadata_padded = metadata_padded.to(device)
    attn_mask = attn_mask.to(device)
    globals_tensor = torch.from_numpy(globals_batch).to(device)

    with torch.no_grad():
        deltas = model(
            terrain_padded, wildlife_padded, metadata_padded,
            attn_mask, globals_tensor,
        ).cpu().numpy()

    leaf_scores = np.array([c["current_score"] for c in candidates], dtype=np.float32)
    totals = leaf_scores + deltas
    return int(np.argmax(totals))


def pick_greedy(candidates):
    scores = [c["current_score"] for c in candidates]
    return int(np.argmax(scores))


def pick_random(candidates, rng):
    return rng.randint(0, len(candidates) - 1)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_benchmark(args):
    device = get_device()
    print(f"Device: {device}")
    print(f"Strategy: {args.strategy}")

    model = None
    if args.strategy == "transformer":
        model = CascadiaTransformer(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        ).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        # Handle checkpoints saved with optional 'model' wrapper
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()
        total, _ = model.count_parameters()
        print(f"Loaded {args.checkpoint} ({total:,} params)")

    rng = random.Random(args.seed)

    cmd = [CASCADIA_CLI, str(args.games), "--external-eval"]
    if args.random_seed:
        cmd.append("--random-seed")
    print(f"Running: {' '.join(cmd)}")
    print(f"Playing {args.games} games...")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)

    scores = []
    num_evals = 0
    t0 = time.time()

    try:
        while True:
            msg_type, payload = read_frame(proc.stdout)
            if msg_type is None:
                break
            if msg_type == MSG_EVAL:
                num_candidates = payload[0]
                offset = 1
                candidates = []
                for _ in range(num_candidates):
                    offset, c = parse_candidate(payload, offset)
                    candidates.append(c)
                if args.strategy == "transformer":
                    idx = pick_transformer_batch(model, device, candidates)
                elif args.strategy == "greedy":
                    idx = pick_greedy(candidates)
                elif args.strategy == "random":
                    idx = pick_random(candidates, rng)
                else:
                    raise ValueError(f"Unknown strategy: {args.strategy}")
                write_pick(proc.stdin, idx)
                num_evals += 1
            elif msg_type == MSG_DONE:
                scores.append(struct.unpack("<H", payload)[0])
            elif msg_type == MSG_FINAL:
                break
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)

    elapsed = time.time() - t0

    print(f"\n{'═' * 50}")
    print(f"Benchmark: {len(scores)} games in {elapsed:.1f}s "
          f"({len(scores)/max(elapsed,1):.2f} g/s)")
    print(f"  {num_evals} evaluation rounds")
    if scores:
        s = np.array(scores)
        print(f"\n  Mean:    {s.mean():.2f}")
        print(f"  Median:  {int(np.median(s))}")
        print(f"  P10:     {int(np.percentile(s, 10))}")
        print(f"  P90:     {int(np.percentile(s, 90))}")
        print(f"  Min/Max: {s.min()}/{s.max()}")
        print(f"  Stddev:  {s.std():.2f}")


def main():
    p = argparse.ArgumentParser(description="Transformer gameplay benchmark")
    p.add_argument("--checkpoint", default="transformer_v1_50k.pt")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--strategy", default="transformer",
                   choices=["transformer", "greedy", "random"])
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-seed", action="store_true")
    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
