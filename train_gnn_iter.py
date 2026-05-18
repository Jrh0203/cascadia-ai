"""Iterative GNN self-play training loop.

Each iter:
    1. Generate N games of GNN-guided self-play → TILE file
    2. Train GNN on fresh data (warm-starting from previous iter's weights)
    3. Benchmark on ~30 games
    4. Repeat

Expected per-iter wall time (on MPS, 50K games, 20 epochs):
    - Self-play (via external-eval): ~12-15 min
    - Training: ~35-40 min (at 110s/epoch)
    - Bench: ~10s
    = ~50 min/iter

Usage:
    # Wait for current iter-1 training to finish, then run iters 2-6
    python3 train_gnn_iter.py --wait-for-pid <PID> \
        --init-weights gnn_v2_50k.pt --iterations 5 --games 50000 --epochs 20
"""

import argparse
import os
import subprocess
import sys
import time


def wait_for_pid(pid, poll_sec=30):
    """Block until the given PID exits."""
    if pid == 0:
        return
    print(f"Waiting for PID {pid} to finish...")
    while True:
        r = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
        if r.returncode != 0:
            print(f"PID {pid} done. Starting iterations.")
            return
        time.sleep(poll_sec)


def run(cmd, description=""):
    print(f"\n{'━' * 60}")
    if description:
        print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'━' * 60}")
    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed = time.time() - t0
    print(f"  → exit={rc}, {elapsed:.0f}s")
    return rc == 0


def main():
    p = argparse.ArgumentParser(description="Iterative GNN self-play training")
    p.add_argument("--init-weights", required=True, help="Starting .pt checkpoint")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--games", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.0005)  # lower than initial (warm-start)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bench-games", type=int, default=30)
    p.add_argument("--run-name", default="gnn_selfplay")
    p.add_argument("--wait-for-pid", type=int, default=0,
                   help="Block until this PID exits before starting iter 2")
    p.add_argument("--iter-offset", type=int, default=1,
                   help="First iter number (defaults to 2 since iter 1 = warm-start input)")
    p.add_argument("--sp-workers", type=int, default=8,
                   help="Parallel self-play workers (default 8). 1 = serial.")
    args = p.parse_args()

    if args.wait_for_pid:
        wait_for_pid(args.wait_for_pid)

    current_weights = args.init_weights
    all_bench_results = []

    total_start = time.time()
    for step in range(1, args.iterations + 1):
        iter_num = step + args.iter_offset  # default: starts at iter 2
        print(f"\n{'═' * 60}")
        print(f"  ITER {iter_num}  (step {step}/{args.iterations})")
        print(f"{'═' * 60}")

        iter_start = time.time()
        sp_out = f"{args.run_name}_iter{iter_num}_selfplay.bin"
        iter_weights = f"{args.run_name}_iter{iter_num}.pt"
        train_log = f"{args.run_name}_iter{iter_num}_train.log"

        # 1. Self-play using current GNN
        sp_cmd = [
            sys.executable, "-u", "selfplay_gnn.py",
            "--checkpoint", current_weights,
            "--games", str(args.games),
            "--num-workers", str(args.sp_workers),
            "--out", sp_out,
            "--hidden", str(args.hidden),
            "--n-layers", str(args.n_layers),
            "--random-seed",
        ]
        if not run(sp_cmd, f"[iter {iter_num}] Self-play with {current_weights}"):
            print("Self-play failed, stopping.")
            return 1

        # 2. Train
        train_cmd = [
            sys.executable, "-u", "train_cnn.py",
            "--samples", sp_out,
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--hidden", str(args.hidden),
            "--n-layers", str(args.n_layers),
            "--batch-size", str(args.batch_size),
            "--init-weights", current_weights,
            "--out", iter_weights,
        ]
        if not run(train_cmd, f"[iter {iter_num}] Train → {iter_weights}"):
            print("Training failed, stopping.")
            return 1

        # 3. Benchmark (quick)
        bench_cmd = [
            sys.executable, "-u", "bench_gnn.py",
            "--checkpoint", iter_weights,
            "--games", str(args.bench_games),
            "--strategy", "gnn",
            "--hidden", str(args.hidden),
            "--n-layers", str(args.n_layers),
        ]
        if not run(bench_cmd, f"[iter {iter_num}] Benchmark"):
            print("Bench failed, continuing anyway.")

        current_weights = iter_weights
        iter_elapsed = time.time() - iter_start
        print(f"\n  Iter {iter_num} complete in {iter_elapsed:.0f}s ({iter_elapsed/60:.1f} min)")

    total_elapsed = time.time() - total_start
    print(f"\n{'═' * 60}")
    print(f"  ALL DONE in {total_elapsed/60:.1f} min — final weights: {current_weights}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    sys.exit(main())
