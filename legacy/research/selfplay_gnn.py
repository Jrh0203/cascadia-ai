"""Generate GNN-guided self-play data in TILE format.

Spawns `cascadia-cli --external-eval N` and drives move selection via the
loaded GNN. Records every player-0 afterstate + the corresponding delta
(final - afterstate_score) as a training sample.

The resulting file is in the same TILE binary format as
`cascadia-cli --tile-token-selfplay`, so `train_cnn.py --samples <file>`
consumes it directly.

Supports parallel workers via `--num-workers N` — each worker runs an
independent Python process (its own GNN on MPS + its own Rust subprocess),
producing ~linear speedup up to GPU saturation (typically 4-8×).
"""

import argparse
import multiprocessing as mp
import os
import struct
import subprocess
import sys
import time

import numpy as np
import torch

from train_cnn import HexGNN, NODE_FEATURES
from bench_gnn import (
    MSG_EVAL, MSG_DONE, MSG_FINAL,
    read_frame, write_pick, parse_candidate, pick_gnn,
)


CASCADIA_CLI = "./target/release/cascadia-cli"


def write_tile_tokens(path, samples):
    """Write accumulated samples in TILE format (matches Rust write_tile_token_samples).

    samples: iterable of (candidate_dict, target_delta)
    """
    with open(path, "wb") as f:
        f.write(b"TILE")
        for cand, target in samples:
            f.write(bytes([cand["num_tiles"]]))
            for terrain, wildlife, allowed, flags, q, r in cand["tiles"]:
                f.write(bytes(terrain))                     # 6 bytes
                f.write(bytes([wildlife, allowed, flags]))  # 3 bytes
                f.write(struct.pack("bb", q, r))            # 2 bytes (i8)
            f.write(cand["global_bytes"])                   # 45 bytes
            f.write(struct.pack("<f", float(target)))       # 4 bytes


def merge_tile_files(input_paths, out_path):
    """Concatenate multiple TILE files (strip magic from all but the first)."""
    total_bytes = 0
    with open(out_path, "wb") as out:
        for i, p in enumerate(input_paths):
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                data = f.read()
            if i == 0:
                out.write(data)
            else:
                # Skip 4-byte "TILE" magic
                out.write(data[4:])
            total_bytes += len(data)
    return total_bytes


def worker_run(worker_id, games, seed_offset, checkpoint, hidden, n_layers,
               out_path, random_seed, log_every=500):
    """Worker entry point: runs self-play in its own process.

    Writes to `out_path` (worker-specific) and returns its summary dict via a queue.
    """
    # Reimport torch in the subprocess (multiprocessing spawn on macOS)
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load GNN in this worker's memory space
    model = HexGNN(
        node_in=NODE_FEATURES, hidden=hidden, n_layers=n_layers, global_dim=53,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    # Launch cascadia-cli with a seed offset
    cmd = [CASCADIA_CLI, str(games), "--external-eval"]
    if random_seed:
        cmd.append("--random-seed")
    env = os.environ.copy()
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)

    # Redirect subprocess stderr to /dev/null so parallel workers don't flood
    # the terminal with overlapping game logs.
    dev_null = open(os.devnull, "wb")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=dev_null, env=env,
    )

    game_snapshots = []
    game_targets = []
    current_game_moves = []
    game_idx = 0
    t0 = time.time()
    last_log = t0

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

                idx = pick_gnn(model, device, candidates)
                write_pick(proc.stdin, idx)
                current_game_moves.append(candidates[idx])

            elif msg_type == MSG_DONE:
                final_score = struct.unpack("<H", payload)[0]
                for cand in current_game_moves:
                    delta = float(final_score) - cand["current_score"]
                    game_snapshots.append(cand)
                    game_targets.append(delta)
                current_game_moves = []
                game_idx += 1

                now = time.time()
                if game_idx % log_every == 0 or now - last_log > 30:
                    rate = game_idx / max(now - t0, 1e-3)
                    remaining = (games - game_idx) / max(rate, 0.01)
                    print(f"  [w{worker_id}] {game_idx}/{games} games "
                          f"({rate:.1f}/s, eta {remaining:.0f}s)", flush=True)
                    last_log = now

            elif msg_type == MSG_FINAL:
                break
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        dev_null.close()

    elapsed = time.time() - t0
    paired = list(zip(game_snapshots, game_targets))
    write_tile_tokens(out_path, paired)
    size_mb = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
    print(f"  [w{worker_id}] DONE: {game_idx} games, {len(paired)} samples, "
          f"{size_mb:.1f} MB in {elapsed:.1f}s", flush=True)
    return {"worker_id": worker_id, "games": game_idx, "samples": len(paired), "elapsed": elapsed}


def run_selfplay_parallel(args):
    """Run N parallel workers, each handling a slice of the games."""
    num_workers = max(1, args.num_workers)
    games_per_worker = args.games // num_workers
    leftover = args.games - games_per_worker * num_workers
    # Give leftover games to the last worker
    worker_games = [games_per_worker] * num_workers
    if leftover > 0:
        worker_games[-1] += leftover

    print(f"Parallel self-play: {num_workers} workers × {games_per_worker} games "
          f"(last worker +{leftover}) = {args.games} total")
    print(f"Model: {args.checkpoint}")

    out_base = args.out
    if out_base.endswith(".bin"):
        out_base = out_base[:-4]
    worker_outs = [f"{out_base}.w{i:02d}.bin" for i in range(num_workers)]

    t_start = time.time()

    # Use spawn to avoid fork/MPS interactions on macOS
    ctx = mp.get_context("spawn")
    procs = []
    for i in range(num_workers):
        p = ctx.Process(
            target=worker_run,
            kwargs=dict(
                worker_id=i,
                games=worker_games[i],
                seed_offset=i * 10_000_000,  # non-overlapping seed ranges
                checkpoint=args.checkpoint,
                hidden=args.hidden,
                n_layers=args.n_layers,
                out_path=worker_outs[i],
                random_seed=args.random_seed,
            ),
        )
        p.start()
        procs.append(p)

    # Wait for all workers
    for p in procs:
        p.join()

    # Check exit codes
    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        print(f"WARNING: {len(failed)} workers exited non-zero: {failed}")

    # Merge outputs
    total_bytes = merge_tile_files(worker_outs, args.out)
    elapsed = time.time() - t_start
    size_mb = os.path.getsize(args.out) / 1e6

    # Cleanup worker files (keep merged result)
    for p in worker_outs:
        if os.path.exists(p):
            os.remove(p)

    print(f"\nMerged → {args.out} ({size_mb:.1f} MB) in {elapsed:.1f}s total")
    print(f"Throughput: {args.games / elapsed:.1f} games/sec "
          f"({num_workers}× parallelism)")


def run_selfplay_single(args):
    """Single-worker path (keeps old behavior for --num-workers 1)."""
    worker_run(
        worker_id=0,
        games=args.games,
        seed_offset=0,
        checkpoint=args.checkpoint,
        hidden=args.hidden,
        n_layers=args.n_layers,
        out_path=args.out,
        random_seed=args.random_seed,
    )


def main():
    p = argparse.ArgumentParser(description="GNN-guided self-play → TILE training data")
    p.add_argument("--checkpoint", required=True, help="GNN .pt checkpoint to drive moves")
    p.add_argument("--games", type=int, default=50000, help="Total games to play")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Parallel workers (default 8). Use 1 for serial.")
    p.add_argument("--out", required=True, help="Output TILE file")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--random-seed", action="store_true")
    args = p.parse_args()

    if args.num_workers <= 1:
        run_selfplay_single(args)
    else:
        run_selfplay_parallel(args)


if __name__ == "__main__":
    main()
