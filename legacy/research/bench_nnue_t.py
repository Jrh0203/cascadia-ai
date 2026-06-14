"""Benchmark NNUE-T (transformer-features NNUE) via external-eval.

Same approach as bench_transformer_v2.py but using the NNUE-T model.

Usage:
    python3 bench_nnue_t.py --checkpoint nnue_t.pt --games 100 --random-seed
"""

import argparse
import struct
import subprocess
import time
import random

import numpy as np
import torch

from train_nnue_t import (
    load_checkpoint, GLOBAL_FEATURES, feature_dims,
    NUM_TERRAIN_TYPES, NUM_WILDLIFE_STATES, TERRAIN_EDGES,
    NEIGHBOR_WL_STATES, NEIGHBOR_TR_STATES,
    GRID_DIM, HALF_GRID,
    G_TURN_BINS, G_TOKEN_BINS, G_WL_COUNT_BINS, G_HAB_BINS, G_BAG_BINS, G_TBAG_BINS,
)

# Hex direction order matches Rust: E, NE, NW, W, SW, SE.
# (dq, dr) per direction.
HEX_DIRS = [
    (1, 0),    # E
    (1, -1),   # NE
    (0, -1),   # NW
    (-1, 0),   # W
    (-1, 1),   # SW
    (0, 1),    # SE
]
from bench_gnn import (
    MSG_EVAL, MSG_DONE, MSG_FINAL,
    read_frame, write_pick, parse_candidate,
)

CASCADIA_CLI = "./target/release/cascadia-cli"


def tile_content_indices(terrain, wildlife, allowed, flags,
                         neighbor_wildlife=None, neighbor_terrain=None):
    """Build content-channel indices for one tile.

    If neighbor_wildlife/neighbor_terrain are provided (lists of length 6),
    appends the 78 adjacency channels.
    """
    idxs = []
    for e in range(TERRAIN_EDGES):
        t = terrain[e]
        if 0 <= t < NUM_TERRAIN_TYPES:
            idxs.append(e * NUM_TERRAIN_TYPES + t)
    base = TERRAIN_EDGES * NUM_TERRAIN_TYPES  # 30
    if 0 <= wildlife < NUM_WILDLIFE_STATES:
        idxs.append(base + wildlife)
    base += NUM_WILDLIFE_STATES  # 36
    for b in range(5):
        if (allowed >> b) & 1:
            idxs.append(base + b)
    base += 5  # 41
    if flags & 1:
        idxs.append(base + 0)
    if (flags >> 1) & 1:
        idxs.append(base + 1)
    base += 2  # 43

    if neighbor_wildlife is not None:
        for d in range(TERRAIN_EDGES):
            nw = neighbor_wildlife[d]
            if 0 <= nw < NEIGHBOR_WL_STATES:
                idxs.append(base + d * NEIGHBOR_WL_STATES + nw)
        base += TERRAIN_EDGES * NEIGHBOR_WL_STATES  # 85
    if neighbor_terrain is not None:
        for d in range(TERRAIN_EDGES):
            nt = neighbor_terrain[d]
            if 0 <= nt < NEIGHBOR_TR_STATES:
                idxs.append(base + d * NEIGHBOR_TR_STATES + nt)
    return idxs


def compute_adjacency(tiles):
    """Given list of (terrain, wildlife, allowed, flags, q, r), return
    [(neighbor_wildlife[6], neighbor_terrain[6])] per tile.

    neighbor_wildlife code: 0=no tile, 1-5=wildlife type, 6=tile-no-wildlife.
    neighbor_terrain code: 0=no tile, 1-5=terrain on neighbor's edge facing back.
    """
    pos_to_tile = {(q, r): t for t in tiles for _, _, _, _, q, r in [t]}
    adj_per_tile = []
    for terrain, wildlife, allowed, flags, q, r in tiles:
        nw = [0] * 6
        nt = [0] * 6
        for d in range(6):
            dq, dr = HEX_DIRS[d]
            n_pos = (q + dq, r + dr)
            if n_pos in pos_to_tile:
                n_terrain, n_wildlife, _, _, _, _ = pos_to_tile[n_pos]
                # Wildlife code: 0=no tile (handled), 1-5=type, 6=tile-no-wildlife
                if 1 <= n_wildlife <= 5:
                    nw[d] = n_wildlife
                else:
                    nw[d] = 6  # tile-no-wildlife
                # Terrain on neighbor's edge facing back at us = (d+3)%6
                back_dir = (d + 3) % 6
                t_val = n_terrain[back_dir]
                if 0 <= t_val < NUM_TERRAIN_TYPES:
                    nt[d] = t_val + 1  # shift so 0=no tile, 1-5=terrain
        adj_per_tile.append((nw, nt))
    return adj_per_tile


def candidate_features(candidate, include_adjacency=False):
    """Extract sparse feature index list (np.int64) for one candidate."""
    cc, axf, total_ax, nf, q_off, r_off, s_off, g_off = feature_dims(include_adjacency)
    feats = set()
    tiles = candidate["tiles"]
    if include_adjacency:
        adj = compute_adjacency(tiles)
    else:
        adj = [(None, None)] * len(tiles)

    for (terrain, wildlife, allowed, flags, q, r), (nw, nt) in zip(tiles, adj):
        s = -q - r
        qb = max(0, min(GRID_DIM - 1, q + HALF_GRID))
        rb = max(0, min(GRID_DIM - 1, r + HALF_GRID))
        sb = max(0, min(GRID_DIM - 1, s + HALF_GRID))
        for c in tile_content_indices(terrain, wildlife, allowed, flags, nw, nt):
            feats.add(q_off + qb * cc + c)
            feats.add(r_off + rb * cc + c)
            feats.add(s_off + sb * cc + c)

    g = candidate["global_bytes"]
    offset = g_off
    feats.add(offset + min(int(g[0]), G_TURN_BINS - 1)); offset += G_TURN_BINS
    feats.add(offset + min(int(g[1]), G_TOKEN_BINS - 1)); offset += G_TOKEN_BINS
    for i in range(5):
        feats.add(offset + min(int(g[2 + i]), G_WL_COUNT_BINS - 1)); offset += G_WL_COUNT_BINS
    for i in range(5):
        feats.add(offset + min(int(g[7 + i]), G_HAB_BINS - 1)); offset += G_HAB_BINS
    for i in range(5):
        feats.add(offset + min(int(g[12 + i]), G_BAG_BINS - 1)); offset += G_BAG_BINS
    for i in range(5):
        feats.add(offset + min(int(g[17 + i]), G_HAB_BINS - 1)); offset += G_HAB_BINS
    for i in range(4):
        v = int(g[22 + i])
        if 0 <= v < NUM_TERRAIN_TYPES:
            feats.add(offset + v)
        offset += NUM_TERRAIN_TYPES
    for i in range(4):
        v = int(g[26 + i])
        if 0 <= v < NUM_TERRAIN_TYPES:
            feats.add(offset + v)
        offset += NUM_TERRAIN_TYPES
    for i in range(4):
        v = int(g[30 + i])
        if 0 <= v < NUM_WILDLIFE_STATES:
            feats.add(offset + v)
        offset += NUM_WILDLIFE_STATES
    for i in range(5):
        feats.add(offset + min(int(g[34 + i]), G_TBAG_BINS - 1)); offset += G_TBAG_BINS
    for i in range(5):
        feats.add(offset + min(int(g[39 + i]), G_TBAG_BINS - 1)); offset += G_TBAG_BINS
    if g[44]:
        feats.add(offset)
    return np.array(sorted(feats), dtype=np.int64)


def pick_nnue_t_batch(model, device, candidates):
    """Score all candidates in one forward pass."""
    include_adj = model.include_adjacency
    feat_idxs_list = []
    samp_idxs_list = []
    for i, c in enumerate(candidates):
        feats = candidate_features(c, include_adjacency=include_adj)
        feat_idxs_list.append(feats)
        samp_idxs_list.append(np.full(len(feats), i, dtype=np.int64))
    feat_idxs = torch.from_numpy(np.concatenate(feat_idxs_list)).to(device)
    samp_idxs = torch.from_numpy(np.concatenate(samp_idxs_list)).to(device)
    with torch.no_grad():
        deltas = model.forward_sparse(feat_idxs, samp_idxs, len(candidates)).cpu().numpy()
    leaf_scores = np.array([c["current_score"] for c in candidates], dtype=np.float32)
    return int(np.argmax(leaf_scores + deltas))


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
    if args.strategy == "nnue_t":
        model = load_checkpoint(args.checkpoint, device).to(device)
        model.eval()
        print(f"Loaded {args.checkpoint} ({model.count_parameters():,} params, "
              f"include_adjacency={model.include_adjacency})")

    rng = random.Random(args.seed)
    cmd = [CASCADIA_CLI, str(args.games), "--external-eval"]
    if args.random_seed:
        cmd.append("--random-seed")
    print(f"Running: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
    scores = []
    num_evals = 0
    t0 = time.time()
    try:
        while True:
            mt, payload = read_frame(proc.stdout)
            if mt is None:
                break
            if mt == MSG_EVAL:
                nc = payload[0]
                offset = 1
                cands = []
                for _ in range(nc):
                    offset, c = parse_candidate(payload, offset)
                    cands.append(c)
                if args.strategy == "nnue_t":
                    idx = pick_nnue_t_batch(model, device, cands)
                elif args.strategy == "greedy":
                    idx = pick_greedy(cands)
                elif args.strategy == "random":
                    idx = pick_random(cands, rng)
                else:
                    raise ValueError(args.strategy)
                write_pick(proc.stdin, idx)
                num_evals += 1
            elif mt == MSG_DONE:
                scores.append(struct.unpack("<H", payload)[0])
            elif mt == MSG_FINAL:
                break
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)

    elapsed = time.time() - t0
    if not scores:
        print("ERROR: no scores")
        return
    arr = np.array(scores)
    print(f"\n=== {args.strategy} ({len(scores)} games, {elapsed:.1f}s, {num_evals} evals) ===")
    print(f"  Mean:   {arr.mean():.2f}")
    print(f"  Median: {np.median(arr):.1f}")
    print(f"  Std:    {arr.std():.2f}")
    print(f"  P10:    {np.percentile(arr, 10):.1f}")
    print(f"  P90:    {np.percentile(arr, 90):.1f}")
    print(f"  Min/Max: {arr.min()} / {arr.max()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="nnue_t.pt")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--strategy", default="nnue_t",
                   choices=["nnue_t", "greedy", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-seed", action="store_true")
    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
