"""Benchmark a trained v2 transformer (cell-position embedding) by playing games.

Uses the external-eval protocol in `cascadia-cli --external-eval`. The 11-byte
per-tile format already contains everything v2 needs (terrain, wildlife,
allowed_mask, flags, q, r) — adjacency is not needed because we use a learned
per-cell positional embedding and let attention discover spatial relationships.

Usage:
    python3 bench_transformer_v2.py --checkpoint transformer_v2_pos.pt --games 50

    # Baselines for comparison (matches bench_transformer.py)
    python3 bench_transformer_v2.py --checkpoint X.pt --games 50 --strategy greedy
"""

import argparse
import struct
import subprocess
import time
import random

import numpy as np
import torch

from train_transformer_v2 import (
    CascadiaTransformerV2, load_checkpoint,
    TILE_CONTENT_DIM, TILE_ADJ_DIM, NUM_CELLS, GLOBAL_FEAT_DIM,
    parse_global_features,
    NUM_TERRAIN_TYPES, NUM_WILDLIFE_TYPES, TERRAIN_EDGES,
    cell_id_from_qr,
)
from bench_gnn import (
    MSG_EVAL, MSG_DONE, MSG_FINAL,
    read_frame, write_pick, parse_candidate,
)


CASCADIA_CLI = "./target/release/cascadia-cli"


def candidate_to_v2_features(candidate):
    """Convert a candidate (11-byte tile format) into v2 content + cell_ids.

    Returns:
        content: np.ndarray [num_tiles, TILE_CONTENT_DIM]
        cell_ids: np.ndarray [num_tiles] int64
    """
    num_tiles = candidate["num_tiles"]
    tiles = candidate["tiles"]  # list of (terrain, wildlife, allowed, flags, q, r)

    out = np.zeros((num_tiles, TILE_CONTENT_DIM), dtype=np.float32)
    qs = np.zeros(num_tiles, dtype=np.int8)
    rs = np.zeros(num_tiles, dtype=np.int8)

    for i, (terrain, wildlife, allowed, flags, q, r) in enumerate(tiles):
        offset = 0
        # 6 edges × 5 terrain
        for e in range(TERRAIN_EDGES):
            t = terrain[e]
            if 0 <= t < NUM_TERRAIN_TYPES:
                out[i, offset + t] = 1.0
            offset += NUM_TERRAIN_TYPES
        # wildlife
        if 0 <= wildlife < NUM_WILDLIFE_TYPES:
            out[i, offset + wildlife] = 1.0
        offset += NUM_WILDLIFE_TYPES
        # allowed_mask 5 bits
        for b in range(5):
            out[i, offset + b] = float((allowed >> b) & 1)
        offset += 5
        # flags
        out[i, offset + 0] = float(flags & 1)
        out[i, offset + 1] = float((flags >> 1) & 1)
        offset += 2
        assert offset == TILE_CONTENT_DIM

        qs[i] = q
        rs[i] = r

    cell_ids = cell_id_from_qr(qs, rs).astype(np.int64)
    return out, cell_ids


def pick_v2_batch(model, device, candidates):
    """Score all candidates in one forward pass; return argmax of (current + delta)."""
    B = len(candidates)
    max_tiles = max(c["num_tiles"] for c in candidates) if candidates else 0
    seq_len = max_tiles + 1  # +1 for CLS

    tile_padded = torch.zeros(B, seq_len, TILE_CONTENT_DIM, dtype=torch.float32)
    cell_padded = torch.zeros(B, seq_len, dtype=torch.long)
    attn_mask = torch.ones(B, seq_len, dtype=torch.bool)
    globals_batch = np.zeros((B, GLOBAL_FEAT_DIM), dtype=np.float32)

    for i, c in enumerate(candidates):
        content, cell_ids = candidate_to_v2_features(c)
        n = c["num_tiles"]
        # CLS at 0 (cell_id = NUM_CELLS); tiles at 1..n
        cell_padded[i, 0] = NUM_CELLS
        tile_padded[i, 1:n + 1] = torch.from_numpy(content)
        cell_padded[i, 1:n + 1] = torch.from_numpy(cell_ids)
        attn_mask[i, :n + 1] = False
        globals_batch[i] = parse_global_features(c["global_bytes"])

    tile_padded = tile_padded.to(device)
    cell_padded = cell_padded.to(device)
    attn_mask = attn_mask.to(device)
    globals_tensor = torch.from_numpy(globals_batch).to(device)

    with torch.no_grad():
        deltas = model(tile_padded, cell_padded, attn_mask, globals_tensor).cpu().numpy()

    leaf_scores = np.array([c["current_score"] for c in candidates], dtype=np.float32)
    totals = leaf_scores + deltas
    return int(np.argmax(totals))


def pick_greedy(candidates):
    return int(np.argmax([c["current_score"] for c in candidates]))


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
    if args.strategy == "transformer_v2":
        model = load_checkpoint(args.checkpoint, device).to(device)
        model.eval()
        total, _ = model.count_parameters()
        print(f"Loaded {args.checkpoint} ({total:,} params, "
              f"include_adjacency={model.include_adjacency})")
        if model.include_adjacency:
            raise NotImplementedError(
                "v2 with adjacency cannot use the basic external-eval protocol "
                "(would need rich-format candidate stream). Train without "
                "--include-adjacency or extend the protocol.")

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
                if args.strategy == "transformer_v2":
                    idx = pick_v2_batch(model, device, candidates)
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
    if not scores:
        print("ERROR: no scores recorded")
        return

    arr = np.array(scores)
    print(f"\n=== Results ({len(scores)} games, {elapsed:.1f}s, {num_evals} evals) ===")
    print(f"  Mean:   {arr.mean():.2f}")
    print(f"  Median: {np.median(arr):.1f}")
    print(f"  Std:    {arr.std():.2f}")
    print(f"  Min:    {arr.min()}")
    print(f"  P10:    {np.percentile(arr, 10):.1f}")
    print(f"  P50:    {np.percentile(arr, 50):.1f}")
    print(f"  P90:    {np.percentile(arr, 90):.1f}")
    print(f"  Max:    {arr.max()}")


def main():
    p = argparse.ArgumentParser(description="Benchmark transformer v2 via external-eval")
    p.add_argument("--checkpoint", type=str, default="transformer_v2_pos.pt")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--strategy", type=str, default="transformer_v2",
                   choices=["transformer_v2", "greedy", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-seed", action="store_true",
                   help="Use random seed for game generation (not just RNG)")
    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
