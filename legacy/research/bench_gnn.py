"""Benchmark the GNN by playing actual games with external-eval protocol.

Launches `cascadia-cli --external-eval N`. At each player 0 turn, reads the
candidate moves' afterstate tile tokens from the subprocess's stdout, scores
them with the GNN, and writes back the picked index on stdin.

Protocol (see external-eval in main.rs):
    Rust → Python (stdout):
        0x01 EVAL: u8 num_candidates, for each:
            u8 num_tiles, 11*num_tiles tile bytes, 45 global bytes, f32 current_score
        0x02 DONE: u16 final_score
        0x03 FINAL: empty
    Python → Rust (stdin):
        0x10 PICK: u8 chosen_idx

Usage:
    python3 bench_gnn.py --checkpoint gnn_v2_50k.pt --games 50

    # Baseline: random choice among candidates
    python3 bench_gnn.py --checkpoint gnn_v2_50k.pt --games 50 --strategy random

    # Baseline: pick candidate with highest current_score (pure greedy)
    python3 bench_gnn.py --checkpoint gnn_v2_50k.pt --games 50 --strategy greedy
"""

import argparse
import struct
import subprocess
import sys
import time
import random

import numpy as np
import torch

from train_cnn import HexGNN, NODE_FEATURES, encode_tile_features, build_edge_index


CASCADIA_CLI = "./target/release/cascadia-cli"


# ─── Binary protocol ───

MSG_EVAL = 0x01
MSG_DONE = 0x02
MSG_FINAL = 0x03
MSG_PICK = 0x10


def read_frame(stream):
    """Read one framed message from a binary stream. Returns (type, payload_bytes)."""
    header = stream.read(5)
    if not header or len(header) < 5:
        return None, None
    msg_type = header[0]
    msg_len = struct.unpack_from("<I", header, 1)[0]
    payload = stream.read(msg_len) if msg_len > 0 else b""
    return msg_type, payload


def write_pick(stream, idx):
    """Write a PICK message."""
    stream.write(bytes([MSG_PICK]))
    stream.write(struct.pack("<I", 1))
    stream.write(bytes([idx]))
    stream.flush()


def parse_candidate(payload, offset):
    """Parse one candidate from EVAL payload. Returns (next_offset, dict with fields)."""
    num_tiles = payload[offset]
    offset += 1

    tile_bytes_needed = num_tiles * 11
    tile_bytes = payload[offset:offset + tile_bytes_needed]
    offset += tile_bytes_needed

    # Parse each tile
    tiles = []
    coords = []
    for t in range(num_tiles):
        base = t * 11
        terrain = list(tile_bytes[base:base + 6])
        wildlife = tile_bytes[base + 6]
        allowed = tile_bytes[base + 7]
        flags = tile_bytes[base + 8]
        q = struct.unpack_from("b", tile_bytes, base + 9)[0]
        r = struct.unpack_from("b", tile_bytes, base + 10)[0]
        tiles.append((terrain, wildlife, allowed, flags, q, r))
        coords.append((q, r))

    # 45 global bytes
    global_bytes = payload[offset:offset + 45]
    offset += 45

    # 4 bytes f32 current_score
    current_score = struct.unpack_from("<f", payload, offset)[0]
    offset += 4

    return offset, {
        "num_tiles": num_tiles,
        "tiles": tiles,
        "coords": coords,
        "global_bytes": global_bytes,
        "current_score": current_score,
    }


def encode_candidate_for_gnn(candidate):
    """Turn a parsed candidate into (node_features, edge_index, global_features) tensors."""
    num_tiles = candidate["num_tiles"]
    node_features = np.zeros((num_tiles, NODE_FEATURES), dtype=np.float32)
    for i, (terrain, wildlife, allowed, flags, q, r) in enumerate(candidate["tiles"]):
        node_features[i] = encode_tile_features(terrain, wildlife, allowed, flags, q, r)

    edge_index = build_edge_index(candidate["coords"])

    # Global features: same normalization as train_cnn.py loader
    raw = candidate["global_bytes"]
    gf = np.zeros(53, dtype=np.float32)
    gi = 0
    gf[gi] = raw[0] / 20.0; gi += 1   # turn
    gf[gi] = raw[1] / 8.0;  gi += 1   # tokens
    for i in range(5): gf[gi] = raw[2 + i] / 10.0;  gi += 1
    for i in range(5): gf[gi] = raw[7 + i] / 13.0;  gi += 1
    for i in range(5): gf[gi] = raw[12 + i] / 20.0; gi += 1
    for i in range(5): gf[gi] = raw[17 + i] / 13.0; gi += 1
    for i in range(4): gf[gi] = raw[22 + i] / 5.0;  gi += 1
    for i in range(4): gf[gi] = raw[26 + i] / 5.0;  gi += 1
    for i in range(4): gf[gi] = raw[30 + i] / 5.0;  gi += 1
    for i in range(5): gf[gi] = raw[34 + i] / 29.0; gi += 1
    for i in range(5): gf[gi] = raw[39 + i] / 29.0; gi += 1
    gf[gi] = float(raw[44]); gi += 1  # overflow

    return (
        torch.from_numpy(node_features),
        edge_index,
        torch.from_numpy(gf),
    )


# ─── Move picking strategies ───

def pick_gnn(model, device, candidates):
    """Score all candidates with the GNN and return the index with highest current + predicted_delta."""
    # Batch everything together for efficiency
    node_features_list = []
    edge_index_list = []
    batch_idx_list = []
    global_features_list = []
    node_offset = 0

    for i, c in enumerate(candidates):
        nf, ei, gf = encode_candidate_for_gnn(c)
        n = nf.shape[0]
        node_features_list.append(nf)
        edge_index_list.append(ei + node_offset)
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
        global_features_list.append(gf)
        node_offset += n

    nf = torch.cat(node_features_list, dim=0).to(device)
    ei = torch.cat(edge_index_list, dim=1).to(device) if edge_index_list else torch.zeros((2, 0), dtype=torch.long, device=device)
    bi = torch.cat(batch_idx_list, dim=0).to(device)
    gf = torch.stack(global_features_list, dim=0).to(device)

    with torch.no_grad():
        predicted_deltas = model(nf, ei, bi, gf, [c["num_tiles"] for c in candidates])

    predicted_totals = np.array([
        candidates[i]["current_score"] + predicted_deltas[i].item()
        for i in range(len(candidates))
    ])
    return int(np.argmax(predicted_totals))


def pick_greedy(candidates):
    """Baseline: pick the candidate with highest current_score."""
    scores = [c["current_score"] for c in candidates]
    return int(np.argmax(scores))


def pick_random(candidates, rng):
    """Baseline: random choice."""
    return rng.randint(0, len(candidates) - 1)


# ─── Main benchmark loop ───

def run_benchmark(args):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Strategy: {args.strategy}")

    model = None
    if args.strategy == "gnn":
        model = HexGNN(
            node_in=NODE_FEATURES,
            hidden=args.hidden,
            n_layers=args.n_layers,
            global_dim=53,
        ).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded {args.checkpoint} ({n_params:,} params)")

    rng = random.Random(args.seed)

    # Launch the Rust subprocess
    cmd = [CASCADIA_CLI, str(args.games), "--external-eval"]
    if args.random_seed:
        cmd.append("--random-seed")

    print(f"Running: {' '.join(cmd)}")
    print(f"Playing {args.games} games...")

    # stderr passes through so we see game progress live
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

                # Pick
                if args.strategy == "gnn":
                    idx = pick_gnn(model, device, candidates)
                elif args.strategy == "greedy":
                    idx = pick_greedy(candidates)
                elif args.strategy == "random":
                    idx = pick_random(candidates, rng)
                else:
                    raise ValueError(f"Unknown strategy: {args.strategy}")

                write_pick(proc.stdin, idx)
                num_evals += 1

            elif msg_type == MSG_DONE:
                final_score = struct.unpack("<H", payload)[0]
                scores.append(final_score)

            elif msg_type == MSG_FINAL:
                break

            else:
                print(f"Unknown msg_type: {msg_type:#x}")
                break
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)

    elapsed = time.time() - t0

    # Report
    print(f"\n{'═' * 50}")
    print(f"Benchmark complete: {len(scores)} games in {elapsed:.1f}s")
    print(f"  {num_evals} evaluation rounds ({num_evals / max(elapsed, 1):.1f} per sec)")
    if scores:
        scores_arr = np.array(scores)
        print(f"\n  Mean:    {scores_arr.mean():.2f}")
        print(f"  Median:  {int(np.median(scores_arr))}")
        print(f"  P10:     {int(np.percentile(scores_arr, 10))}")
        print(f"  P90:     {int(np.percentile(scores_arr, 90))}")
        print(f"  Min/Max: {scores_arr.min()}/{scores_arr.max()}")
        print(f"  Stddev:  {scores_arr.std():.2f}")

    return scores


def main():
    p = argparse.ArgumentParser(description="Benchmark GNN via external-eval protocol")
    p.add_argument("--checkpoint", default="gnn_v2_50k.pt")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--strategy", default="gnn", choices=["gnn", "greedy", "random"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-seed", action="store_true")
    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
