"""Benchmark GNN with MCE-style rollouts.

For each player-0 decision:
    1. Rust generates candidates (~15).
    2. For each candidate, Rust runs N rollouts (depth D, greedy policy).
    3. Leaf tile tokens are batched and sent to Python.
    4. Python GNN predicts delta per leaf, averages (leaf_score + delta) across
       rollouts per candidate, and picks the argmax.
    5. Rust applies the picked move, advances to next turn.

Protocol (see `--gnn-mce-bench` in main.rs):
    Rust → Python:
        0x04 MCE_EVAL: u8 num_cands, u16 R, then (num_cands * R) leaves:
            u8 num_tiles, 11*num_tiles tile bytes, 45 global bytes, f32 leaf_score
        0x02 DONE: u16 final_score
        0x03 FINAL: empty
    Python → Rust:
        0x10 PICK: u8 chosen_idx

Usage:
    python3 bench_gnn_mce.py --checkpoint gnn_sp_iter6.pt --games 30 --rollouts 50 --depth 6
"""

import argparse
import struct
import subprocess
import sys
import time

import numpy as np
import torch

from train_cnn import HexGNN, NODE_FEATURES, encode_tile_features, build_edge_index
from bench_gnn import (
    MSG_DONE, MSG_FINAL, MSG_PICK,
    read_frame, write_pick, parse_candidate,
)


CASCADIA_CLI = "./target/release/cascadia-cli"
MSG_MCE_EVAL = 0x04


def parse_mce_eval(payload):
    """Parse an MCE_EVAL payload. Returns (num_candidates, rollouts_per_candidate, leaves).

    leaves: list of num_candidates × rollouts dicts (num_tiles, tiles, coords,
    global_bytes, current_score).
    """
    num_cands = payload[0]
    rollouts = struct.unpack_from("<H", payload, 1)[0]
    offset = 3
    total = num_cands * rollouts
    leaves = []
    for _ in range(total):
        offset, c = parse_candidate(payload, offset)
        leaves.append(c)
    return num_cands, rollouts, leaves


def score_leaves_batch(model, device, leaves):
    """Run GNN inference on a batch of leaves. Returns 1D np array of predicted deltas."""
    if not leaves:
        return np.zeros(0, dtype=np.float32)

    node_features_list = []
    edge_index_list = []
    batch_idx_list = []
    global_features_list = []
    node_offset = 0

    for i, c in enumerate(leaves):
        # Convert leaf to tensors (same as encode_candidate_for_gnn in bench_gnn.py)
        num_tiles = c["num_tiles"]
        if num_tiles == 0:
            # Pad with a single zero node to keep batching alive
            nf = torch.zeros((1, NODE_FEATURES), dtype=torch.float32)
            ei = torch.zeros((2, 0), dtype=torch.long)
            coords = [(0, 0)]
        else:
            nf_np = np.zeros((num_tiles, NODE_FEATURES), dtype=np.float32)
            for ti, (terrain, wildlife, allowed, flags, q, r) in enumerate(c["tiles"]):
                nf_np[ti] = encode_tile_features(terrain, wildlife, allowed, flags, q, r)
            nf = torch.from_numpy(nf_np)
            coords = c["coords"]
            ei = build_edge_index(coords)

        n = nf.shape[0]

        # Global features (normalize)
        raw = c["global_bytes"]
        gf = np.zeros(53, dtype=np.float32)
        gi = 0
        gf[gi] = raw[0] / 20.0; gi += 1
        gf[gi] = raw[1] / 8.0;  gi += 1
        for k in range(5): gf[gi] = raw[2 + k] / 10.0;  gi += 1
        for k in range(5): gf[gi] = raw[7 + k] / 13.0;  gi += 1
        for k in range(5): gf[gi] = raw[12 + k] / 20.0; gi += 1
        for k in range(5): gf[gi] = raw[17 + k] / 13.0; gi += 1
        for k in range(4): gf[gi] = raw[22 + k] / 5.0;  gi += 1
        for k in range(4): gf[gi] = raw[26 + k] / 5.0;  gi += 1
        for k in range(4): gf[gi] = raw[30 + k] / 5.0;  gi += 1
        for k in range(5): gf[gi] = raw[34 + k] / 29.0; gi += 1
        for k in range(5): gf[gi] = raw[39 + k] / 29.0; gi += 1
        gf[gi] = float(raw[44]); gi += 1

        node_features_list.append(nf)
        edge_index_list.append(ei + node_offset)
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
        global_features_list.append(torch.from_numpy(gf))
        node_offset += n

    nf = torch.cat(node_features_list, dim=0).to(device)
    ei = (torch.cat(edge_index_list, dim=1).to(device)
          if any(e.shape[1] > 0 for e in edge_index_list) else
          torch.zeros((2, 0), dtype=torch.long, device=device))
    bi = torch.cat(batch_idx_list, dim=0).to(device)
    gf = torch.stack(global_features_list, dim=0).to(device)

    with torch.no_grad():
        deltas = model(nf, ei, bi, gf, [c["num_tiles"] for c in leaves])

    return deltas.cpu().numpy()


def pick_mce(model, device, num_cands, rollouts, leaves):
    """Aggregate leaf predictions per candidate and return argmax index.

    For each candidate: avg(leaf_score + predicted_delta) over its rollouts.
    """
    deltas = score_leaves_batch(model, device, leaves)
    leaf_scores = np.array([leaves[i]["current_score"] for i in range(len(leaves))], dtype=np.float32)
    per_leaf_pred = leaf_scores + deltas  # predicted final score from this rollout leaf

    # Average over rollouts per candidate
    per_leaf_pred = per_leaf_pred.reshape(num_cands, rollouts)
    per_cand_avg = per_leaf_pred.mean(axis=1)
    return int(np.argmax(per_cand_avg)), per_cand_avg


def run_benchmark(args):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HexGNN(
        node_in=NODE_FEATURES, hidden=args.hidden, n_layers=args.n_layers, global_dim=53,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.checkpoint} ({n_params:,} params)")

    cmd = [
        CASCADIA_CLI, str(args.games), "--gnn-mce-bench",
        "--rollouts", str(args.rollouts),
        "--depth", str(args.depth),
    ]
    if args.random_seed:
        cmd.append("--random-seed")
    print(f"Running: {' '.join(cmd)}")
    print(f"Config: {args.games} games, {args.rollouts} rollouts/cand, depth {args.depth}")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)

    scores = []
    num_decisions = 0
    total_leaves = 0
    t0 = time.time()

    try:
        while True:
            msg_type, payload = read_frame(proc.stdout)
            if msg_type is None:
                break

            if msg_type == MSG_MCE_EVAL:
                num_cands, rollouts, leaves = parse_mce_eval(payload)
                idx, _avgs = pick_mce(model, device, num_cands, rollouts, leaves)
                write_pick(proc.stdin, idx)
                num_decisions += 1
                total_leaves += len(leaves)

            elif msg_type == MSG_DONE:
                final_score = struct.unpack("<H", payload)[0]
                scores.append(final_score)

            elif msg_type == MSG_FINAL:
                break
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)

    elapsed = time.time() - t0

    print(f"\n{'═' * 50}")
    print(f"GNN MCE bench: {len(scores)} games in {elapsed:.1f}s "
          f"({len(scores)/max(elapsed,1):.2f} g/s)")
    print(f"  {num_decisions} decisions, {total_leaves:,} leaves evaluated "
          f"({total_leaves/max(elapsed,1):.0f}/sec)")
    if scores:
        s = np.array(scores)
        print(f"\n  Mean:    {s.mean():.2f}")
        print(f"  Median:  {int(np.median(s))}")
        print(f"  P10:     {int(np.percentile(s, 10))}")
        print(f"  P90:     {int(np.percentile(s, 90))}")
        print(f"  Min/Max: {s.min()}/{s.max()}")
        print(f"  Stddev:  {s.std():.2f}")


def main():
    p = argparse.ArgumentParser(description="GNN MCE(N) rollout benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--rollouts", type=int, default=50,
                   help="Rollouts per candidate per decision (default 50).")
    p.add_argument("--depth", type=int, default=6,
                   help="AI-turns depth for each rollout (default 6).")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--random-seed", action="store_true")
    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
