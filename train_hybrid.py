"""Hybrid ExIt training: NNUE self-play + MCE data + PyTorch training.

Each iteration:
1. Generate 100K self-play games with current NNUE (~10s, parallel)
2. Combine with MCE expert data (our existing 324K samples)
3. Train with PyTorch + Adam for N epochs
4. Save new weights, repeat

Usage:
    python3 train_hybrid.py --iterations 5 --self-play-games 100000 --epochs-per-iter 15

    # With existing MCE data mixed in:
    python3 train_hybrid.py --iterations 5 --mce-samples mce_policy_samples.bin

    # Resume from weights:
    python3 train_hybrid.py --iterations 5 --init-weights nnue_weights_mce93.bin
"""

import argparse
import os
import subprocess
import sys
import time

CASCADIA_CLI = "./target/release/cascadia-cli"


def run_self_play(num_games, weights_path, out_path, epsilon=0.1):
    """Generate self-play data using Rust (fast, parallel)."""
    cmd = [
        CASCADIA_CLI, str(num_games),
        "--self-play",
        "--epsilon", str(epsilon),
        "--out", out_path,
    ]
    if weights_path:
        cmd.extend(["--weights", weights_path])

    print(f"  Generating {num_games} self-play games (epsilon={epsilon})...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    print(f"  Done in {time.time()-t0:.1f}s")
    return True


def merge_samples(files, out_path):
    """Merge multiple MCEP files into one."""
    with open(out_path, 'wb') as out:
        for i, f in enumerate(files):
            if not os.path.exists(f):
                continue
            data = open(f, 'rb').read()
            if i == 0:
                out.write(data)
            else:
                out.write(data[4:])  # skip magic header
    total_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"  Merged {len(files)} files → {out_path} ({total_size / 1e6:.1f} MB)")


def run_pytorch_training(samples_path, epochs, lr, init_weights, out_weights,
                         hidden1=512, hidden2=64, batch_size=4096):
    """Run PyTorch training."""
    cmd = [
        sys.executable, "-u", "train_pytorch.py",
        "--samples", samples_path,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--hidden1", str(hidden1),
        "--hidden2", str(hidden2),
        "--out", out_weights,
        "--no-augment",  # skip augmentation for speed with large self-play data
    ]
    if init_weights and os.path.exists(init_weights):
        cmd.extend(["--init-weights", init_weights])

    print(f"  Training: {epochs} epochs, lr={lr}, {hidden1}→{hidden2}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  Training failed!")
        return False
    print(f"  Training done in {time.time()-t0:.1f}s")
    return True


def run_benchmark(weights_path, num_games=50):
    """Quick benchmark."""
    cmd = [CASCADIA_CLI, str(num_games), "--nnue", "--weights", weights_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Extract mean score from output
    for line in result.stdout.split('\n'):
        if 'Mean:' in line and 'Base' not in line and 'Habitat' not in line:
            print(f"    {line.strip()}")
            break
    for line in result.stdout.split('\n'):
        if 'With Habitat' in line or 'avg bonus' in line:
            print(f"    {line.strip()}")
            break
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description='Hybrid ExIt training')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--self-play-games', type=int, default=100000)
    parser.add_argument('--epochs-per-iter', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--hidden1', type=int, default=512)
    parser.add_argument('--hidden2', type=int, default=64)
    parser.add_argument('--mce-samples', default='mce_policy_samples.bin',
                        help='MCE expert data to mix in')
    parser.add_argument('--init-weights', default=None)
    parser.add_argument('--out', default='nnue_weights_hybrid.bin')
    parser.add_argument('--benchmark-games', type=int, default=50)
    args = parser.parse_args()

    current_weights = args.init_weights
    total_start = time.time()

    print(f"=== Hybrid ExIt Training ===")
    print(f"  Iterations: {args.iterations}")
    print(f"  Self-play games/iter: {args.self_play_games}")
    print(f"  Epochs/iter: {args.epochs_per_iter}")
    print(f"  Architecture: {args.hidden1}→{args.hidden2}")
    print(f"  MCE expert data: {args.mce_samples}")
    print(f"  Init weights: {current_weights or 'fresh'}")
    print()

    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        print(f"{'='*60}")
        print(f"  ITERATION {iteration}/{args.iterations}")
        print(f"{'='*60}")

        # 1. Generate self-play data
        self_play_path = f"self_play_iter{iteration}.bin"
        strategy = "NNUE" if current_weights else "greedy"
        run_self_play(
            args.self_play_games,
            current_weights,
            self_play_path,
            epsilon=args.epsilon,
        )

        # 2. Merge self-play + MCE expert data
        merged_path = f"training_merged_iter{iteration}.bin"
        files_to_merge = [self_play_path]
        if os.path.exists(args.mce_samples):
            files_to_merge.append(args.mce_samples)
        merge_samples(files_to_merge, merged_path)

        # 3. Train
        iter_weights = f"nnue_weights_hybrid_iter{iteration}.bin"
        run_pytorch_training(
            merged_path,
            epochs=args.epochs_per_iter,
            lr=args.lr,
            init_weights=current_weights,
            out_weights=iter_weights,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
        )

        # 4. Benchmark
        print(f"\n  Benchmark (iteration {iteration}):")
        run_benchmark(iter_weights, args.benchmark_games)

        # 5. Update weights for next iteration
        current_weights = iter_weights

        # Copy to final output
        if os.path.exists(iter_weights):
            import shutil
            shutil.copy2(iter_weights, args.out)

        iter_time = time.time() - iter_start
        print(f"\n  Iteration {iteration} complete in {iter_time:.1f}s")
        print()

    total_time = time.time() - total_start
    print(f"{'='*60}")
    print(f"  ALL ITERATIONS COMPLETE in {total_time:.1f}s")
    print(f"  Final weights: {args.out}")
    print(f"{'='*60}")

    # Final benchmark
    print(f"\n  Final benchmark ({args.benchmark_games} games):")
    run_benchmark(args.out, args.benchmark_games)


if __name__ == '__main__':
    main()
