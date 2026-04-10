"""Self-play training loop: MCTS self-play -> joint training -> repeat.

AlphaGo Zero-style iteration:
1. Play games with MCTS using current weights
2. Train both value and policy heads jointly from scratch
3. Benchmark, keep best weights
4. Repeat

Usage:
    python3 train_selfplay.py --iterations 10 --games-per-iter 500 --simulations 100
    python3 train_selfplay.py --iterations 5 --init-weights nnue_weights_hybrid_iter4.bin
"""

import argparse
import os
import subprocess
import time
import shutil


def run_cmd(cmd, description=""):
    """Run a shell command and return stdout."""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        # Print stderr but don't fail (Rust warnings go to stderr)
        for line in result.stderr.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Self-play training loop')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--games-per-iter', type=int, default=500,
                        help='MCTS self-play games per iteration')
    parser.add_argument('--simulations', type=int, default=100,
                        help='MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs per iteration')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for joint training')
    parser.add_argument('--benchmark-games', type=int, default=200,
                        help='Games for benchmarking')
    parser.add_argument('--init-weights', default=None,
                        help='Starting weights (or None for fresh)')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='MCTS temperature for early moves')
    parser.add_argument('--policy-weight', type=float, default=1.0)
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train from scratch (use init-weights only for iteration 1 data collection)')
    parser.add_argument('--start-iter', type=int, default=1,
                        help='Starting iteration (for resuming from checkpoint)')
    args = parser.parse_args()

    best_score = 0.0
    best_weights = args.init_weights
    current_weights = args.init_weights

    # For from-scratch: first iteration uses init weights for MCTS data collection
    # (to get reasonable game quality), but training starts fresh
    from_scratch = not args.init_weights or args.from_scratch

    # Create work directory
    os.makedirs('selfplay_work', exist_ok=True)

    print(f"Self-play training loop")
    print(f"  Iterations: {args.iterations}")
    print(f"  Games/iter: {args.games_per_iter}")
    print(f"  MCTS sims:  {args.simulations}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Init:       {current_weights or 'fresh'}")

    # Initial benchmark
    if current_weights:
        result = run_cmd(
            f"./target/release/cascadia-cli {args.benchmark_games} --nnue "
            f"--weights {current_weights}",
            "Initial NNUE benchmark"
        )
        best_score = parse_mean_score(result.stdout)
        print(f"  Baseline NNUE score: {best_score:.1f}")

    total_start = time.time()

    # Resume support: if start_iter > 1, load last checkpoint as current weights
    if args.start_iter > 1:
        prev = f"selfplay_work/iter{args.start_iter - 1}_weights.bin"
        if os.path.exists(prev):
            current_weights = prev
            print(f"  Resuming from iteration {args.start_iter}, weights: {prev}")

    for iteration in range(args.start_iter, args.iterations + 1):
        iter_start = time.time()
        print(f"\n{'#'*60}")
        print(f"  ITERATION {iteration}/{args.iterations}")
        print(f"{'#'*60}")

        data_path = f"selfplay_work/iter{iteration}_data.bin"
        weights_path = f"selfplay_work/iter{iteration}_weights.bin"

        # Step 1: MCTS self-play data collection
        # For from-scratch: iteration 1 uses init weights for data quality,
        # subsequent iterations use the jointly-trained weights
        collect_weights = current_weights
        if from_scratch and iteration == 1 and args.init_weights:
            collect_weights = args.init_weights
        weights_flag = f"--weights {collect_weights}" if collect_weights else ""
        result = run_cmd(
            f"./target/release/cascadia-cli {args.games_per_iter} --collect-mcts "
            f"{weights_flag} --simulations {args.simulations} "
            f"--temperature {args.temperature} --random-seed "
            f"--out {data_path}",
            f"Step 1: MCTS self-play ({args.games_per_iter} games, {args.simulations} sims)"
        )

        if not os.path.exists(data_path):
            print(f"ERROR: No data file generated at {data_path}")
            break

        data_size = os.path.getsize(data_path)
        print(f"  Data: {data_size / 1e6:.1f} MB")

        # Step 2: Joint training
        # From-scratch: never init from value-only weights, always continue from last iter
        # Fine-tune: init from provided weights on iter 1, continue from last iter after
        init_flag = ""
        if iteration > 1:
            prev_weights = f"selfplay_work/iter{iteration-1}_weights.bin"
            if os.path.exists(prev_weights):
                init_flag = f"--init-weights {prev_weights}"
        elif not from_scratch and args.init_weights:
            init_flag = f"--init-weights {args.init_weights}"
        # else: fresh random init for iteration 1 from-scratch

        result = run_cmd(
            f"python3 train_pytorch.py policy "
            f"--policy-data {data_path} "
            f"{init_flag} "
            f"--epochs {args.epochs} --lr {args.lr} "
            f"--policy-weight {args.policy_weight} "
            f"--joint "
            f"--out {weights_path}",
            f"Step 2: Joint training ({args.epochs} epochs)"
        )

        if not os.path.exists(weights_path):
            print(f"ERROR: No weights file at {weights_path}")
            break

        # Step 3: Benchmark NNUE-only
        result = run_cmd(
            f"./target/release/cascadia-cli {args.benchmark_games} --nnue "
            f"--weights {weights_path}",
            f"Step 3: Benchmark NNUE-only ({args.benchmark_games} games)"
        )
        nnue_score = parse_mean_score(result.stdout)

        # Step 3b: Quick MCTS benchmark
        mcts_bench_games = min(20, args.benchmark_games)
        result = run_cmd(
            f"./target/release/cascadia-cli {mcts_bench_games} --mcts-search "
            f"--simulations {args.simulations} --weights {weights_path}",
            f"Step 3b: Benchmark MCTS ({mcts_bench_games} games)"
        )
        mcts_score = parse_mean_score(result.stdout)

        iter_elapsed = time.time() - iter_start

        # Step 4: Keep best
        improved = nnue_score > best_score
        if improved:
            best_score = nnue_score
            best_weights = weights_path
            shutil.copy2(weights_path, "nnue_weights_selfplay_best.bin")

        current_weights = weights_path

        print(f"\n  --- Iteration {iteration} Summary ---")
        print(f"  NNUE-only:  {nnue_score:.1f} {'NEW BEST' if improved else ''}")
        print(f"  MCTS({args.simulations}): {mcts_score:.1f}")
        print(f"  Best so far: {best_score:.1f}")
        print(f"  Time: {iter_elapsed:.0f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Self-play complete: {args.iterations} iterations in {total_elapsed:.0f}s")
    print(f"Best NNUE score: {best_score:.1f}")
    if best_weights:
        print(f"Best weights: {best_weights}")
        if os.path.exists("nnue_weights_selfplay_best.bin"):
            print(f"Also saved to: nnue_weights_selfplay_best.bin")


def parse_mean_score(output):
    """Extract mean base score from CLI benchmark output."""
    import re
    for line in output.split('\n'):
        if 'Mean:' in line and 'bonus' not in line.lower():
            m = re.search(r'Mean:\s+([\d.]+)', line)
            if m:
                return float(m.group(1))
    return 0.0


if __name__ == '__main__':
    main()
