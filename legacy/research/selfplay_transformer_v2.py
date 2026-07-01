"""Iterative self-play data collection driven by transformer v2.

Spawns `cascadia-cli --external-eval N`, uses a trained transformer to pick
moves at each player-0 decision, and captures every chosen afterstate into
a TIL2 file labeled with `final_game_score - current_score`.

Output is consumable by `train_transformer_v2.py` for the next training iter.

Usage:
    python3 selfplay_transformer_v2.py \\
        --checkpoint transformer_v2_pos.pt \\
        --games 5000 \\
        --out transformer_v2_iter1.bin \\
        --epsilon 0.1   # exploration
"""

import argparse
import struct
import subprocess
import time
import random

import numpy as np
import torch

from train_transformer_v2 import (
    load_checkpoint, NUM_CELLS, GLOBAL_FEAT_DIM, TILE_CONTENT_DIM,
    parse_global_features, cell_id_from_qr,
)
from bench_transformer_v2 import (
    candidate_to_v2_features, pick_v2_batch, get_device,
)
from bench_gnn import (
    MSG_EVAL, MSG_DONE, MSG_FINAL,
    read_frame, write_pick, parse_candidate,
)


CASCADIA_CLI = "./target/release/cascadia-cli"


def write_til2(path, samples):
    """Write samples as TIL2 format (compatible with train_transformer_v2).

    Each sample is (tiles_11byte, global_45byte, target_f32).
    Stored as TIL2 with neighbor fields zeroed (unused by no-adjacency model).
    """
    with open(path, "wb") as f:
        f.write(b"TIL2")
        for tiles, global_bytes, target in samples:
            num_tiles = len(tiles)
            f.write(bytes([num_tiles]))
            for terrain, wildlife, allowed, flags, q, r in tiles:
                f.write(bytes(terrain))                          # 6 bytes
                f.write(bytes([wildlife, allowed, flags]))       # 3 bytes
                f.write(struct.pack("bb", q, r))                 # 2 bytes
                f.write(bytes([0] * 12))                         # 12 zero bytes (neighbor_wl + neighbor_terrain)
            f.write(bytes(global_bytes))                          # 45 bytes
            f.write(struct.pack("<f", float(target)))             # 4 bytes


def pick_with_epsilon(model, device, candidates, epsilon, rng):
    """Argmax with epsilon-random fallback for exploration."""
    if rng.random() < epsilon:
        return rng.randint(0, len(candidates) - 1)
    return pick_v2_batch(model, device, candidates)


def run_selfplay(args):
    device = get_device()
    print(f"Device: {device}")

    model = load_checkpoint(args.checkpoint, device).to(device)
    model.eval()
    total, _ = model.count_parameters()
    print(f"Loaded {args.checkpoint} ({total:,} params, "
          f"include_adjacency={model.include_adjacency})")
    if model.include_adjacency:
        raise NotImplementedError("v2 with adjacency not supported in self-play yet")

    rng = random.Random(args.seed)

    cmd = [CASCADIA_CLI, str(args.games), "--external-eval"]
    if args.random_seed:
        cmd.append("--random-seed")
    print(f"Running: {' '.join(cmd)}")
    print(f"Self-play {args.games} games (epsilon={args.epsilon})...")

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)

    # Per-game state: list of (tiles_11byte, global_bytes, current_score) for
    # each chosen afterstate. When the game ends we get final_score and label
    # them all.
    pending_afterstates = []  # list of (tiles, global_bytes, current_score)
    all_samples = []          # list of (tiles, global_bytes, target_delta)
    scores = []
    num_evals = 0
    games_finished = 0
    t0 = time.time()
    last_report = t0

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

                idx = pick_with_epsilon(model, device, candidates, args.epsilon, rng)
                write_pick(proc.stdin, idx)

                # Capture the chosen afterstate for training
                chosen = candidates[idx]
                pending_afterstates.append((
                    chosen["tiles"],
                    bytes(chosen["global_bytes"]),
                    chosen["current_score"],
                ))
                num_evals += 1

            elif msg_type == MSG_DONE:
                final_score = struct.unpack("<H", payload)[0]
                scores.append(final_score)
                # Label all pending afterstates and add to dataset
                for tiles, global_bytes, current_score in pending_afterstates:
                    delta = float(final_score) - float(current_score)
                    all_samples.append((tiles, global_bytes, delta))
                pending_afterstates = []
                games_finished += 1

                now = time.time()
                if now - last_report > 30.0:
                    elapsed = now - t0
                    rate = games_finished / elapsed if elapsed > 0 else 0
                    print(f"  {games_finished}/{args.games} games  "
                          f"({rate:.1f} g/s, {len(all_samples)} samples, "
                          f"avg score so far {np.mean(scores):.1f})")
                    last_report = now

            elif msg_type == MSG_FINAL:
                break
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)

    elapsed = time.time() - t0

    if not scores:
        print("ERROR: no games completed")
        return

    arr = np.array(scores)
    print(f"\n=== Self-play complete ({len(scores)} games, {elapsed:.1f}s) ===")
    print(f"  Mean: {arr.mean():.2f}  Median: {np.median(arr):.1f}  P90: {np.percentile(arr, 90):.1f}")
    print(f"  Samples collected: {len(all_samples)}")

    write_til2(args.out, all_samples)
    print(f"  Wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description="Transformer v2 self-play data generation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--out", required=True)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-seed", action="store_true")
    args = p.parse_args()
    run_selfplay(args)


if __name__ == "__main__":
    main()
