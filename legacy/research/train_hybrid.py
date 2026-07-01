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

# Force line-buffered stdout so logs flush immediately when piped to tee/files.
# Without this, multi-line subprocess output (e.g. self-play stats) sits in
# Python's stdout buffer for minutes before showing up.
sys.stdout.reconfigure(line_buffering=True)

CASCADIA_CLI = "./target/release/cascadia-cli"


def run_self_play(num_games, weights_path, out_path, epsilon=0.1, top_pct=100.0,
                  aux_targets=False, opp_weights=None, temperature=None):
    """Generate self-play data using Rust (fast, parallel).

    - aux_targets: write MCV3 format with aux + target_wildlife (recommended default).
    - opp_weights: path to opponent NNUE weights — players 1-3 use this net. Used for
      cross-training (two runs with --opp-weights pointing at each other).
    - temperature: if set, replaces ε-greedy with softmax sampling at the given T.
    """
    cmd = [
        CASCADIA_CLI, str(num_games),
        "--self-play",
        "--epsilon", str(epsilon),
        "--out", out_path,
    ]
    if weights_path:
        cmd.extend(["--weights", weights_path])
    if top_pct < 100.0:
        cmd.extend(["--top-pct", str(top_pct)])
    if aux_targets:
        cmd.append("--aux-targets")
    if opp_weights:
        cmd.extend(["--opp-weights", opp_weights])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])

    extras = []
    if opp_weights: extras.append(f"opp={opp_weights}")
    if temperature is not None: extras.append(f"T={temperature}")
    extra_str = f" [{' '.join(extras)}]" if extras else ""
    print(f"  Generating {num_games} self-play games (epsilon={epsilon}){extra_str}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    print(f"  Done in {time.time()-t0:.1f}s")
    return True


def run_self_play_modal(num_games, weights_path, out_path, epsilon=0.1,
                        aux_targets=False, temperature=None, num_workers=20,
                        max_retries=2):
    """Generate self-play data on Modal workers. Retries on failure."""
    games_per_worker = num_games // num_workers
    cmd = [
        sys.executable, "-m", "modal", "run", "modal_collect.py::self_play",
        "--num-workers", str(num_workers),
        "--games-per-worker", str(games_per_worker),
        "--epsilon", str(epsilon),
        "--out", out_path,
    ]
    if weights_path and os.path.exists(weights_path):
        cmd.extend(["--weights", weights_path])
    if aux_targets:
        cmd.append("--aux-targets")
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])

    for attempt in range(1, max_retries + 1):
        label = f" (attempt {attempt}/{max_retries})" if max_retries > 1 else ""
        print(f"  Modal self-play: {num_workers} workers × {games_per_worker} games = {num_games}{label}")
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            print(f"  Modal self-play done in {time.time()-t0:.1f}s")
            break
        print(f"  WARNING: Modal self-play failed (exit {result.returncode})")
        if attempt < max_retries:
            print(f"  Retrying in 10s...")
            time.sleep(10)
    else:
        print(f"  ERROR: Modal self-play failed after {max_retries} attempts, falling back to local")
        return run_self_play(num_games, weights_path, out_path, epsilon=epsilon,
                             aux_targets=aux_targets, temperature=temperature)
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
                         hidden1=512, hidden2=64, batch_size=4096, optimizer='sgd',
                         use_aux=False, aux_bear_weight=0.3, aux_salmon_weight=0.3,
                         split_value_head=False):
    """Run PyTorch training."""
    cmd = [
        sys.executable, "-u", "train_pytorch.py",
        "value",  # subcommand
        "--samples", samples_path,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--hidden1", str(hidden1),
        "--hidden2", str(hidden2),
        "--optimizer", optimizer,
        "--out", out_weights,
        "--no-augment",  # skip augmentation for speed with large self-play data
    ]
    if use_aux or split_value_head:
        cmd.append("--use-aux")
        cmd.extend(["--aux-bear-weight", str(aux_bear_weight)])
        cmd.extend(["--aux-salmon-weight", str(aux_salmon_weight)])
    if split_value_head:
        cmd.append("--split-value-head")
    if init_weights and os.path.exists(init_weights):
        cmd.extend(["--init-weights", init_weights])

    tags = []
    if split_value_head: tags.append("split")
    if use_aux and not split_value_head: tags.append("aux")
    tag_str = f" [{','.join(tags)}]" if tags else ""
    print(f"  Training: {epochs} epochs, lr={lr}, {hidden1}→{hidden2}{tag_str}")
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
    parser.add_argument('--epsilon-end', type=float, default=None,
                        help='If set, anneal ε from --epsilon to this value linearly across iterations')
    parser.add_argument('--top-pct', type=float, default=100.0,
                        help='Keep only top N%% of self-play games by score (default: 100, all games)')
    parser.add_argument('--hidden1', type=int, default=512)
    parser.add_argument('--hidden2', type=int, default=64)
    parser.add_argument('--mce-samples', default='mce_policy_samples.bin',
                        help='MCE expert data to mix in (use "none" to skip)')
    parser.add_argument('--no-mce', action='store_true',
                        help='Skip MCE data, train on self-play only')
    parser.add_argument('--init-weights', default=None)
    parser.add_argument('--out', default='nnue_weights_hybrid.bin')
    parser.add_argument('--benchmark-games', type=int, default=50)
    parser.add_argument('--iter-offset', type=int, default=0,
                        help='Add this to iteration numbers (avoids overwriting existing iter files)')
    parser.add_argument('--iter-prefix', default='nnue_weights_hybrid_iter',
                        help='Prefix for per-iteration weight files')
    parser.add_argument('--aux-targets', action='store_true',
                        help='Generate MCV3 self-play with aux targets and train multi-task '
                             '(now default for all new pipelines; flag retained for clarity)')
    parser.add_argument('--aux-bear-weight', type=float, default=0.3)
    parser.add_argument('--aux-salmon-weight', type=float, default=0.3)
    parser.add_argument('--split-value-head', action='store_true',
                        help='Use v5 split value head architecture (wildlife + habitat+tokens, '
                             '1:1 sum at inference). Requires self-play data in MCV3 format.')
    parser.add_argument('--opp-weights', default=None,
                        help='Opponent NNUE weights for cross-training: players 1-3 in self-play '
                             'use this net instead of the training net. Useful for running two '
                             'trainings in tandem where each uses the other as opponent.')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Enable softmax sampling in self-play with this initial temperature. '
                             'If set, replaces ε-greedy; anneal to --temperature-end across iterations.')
    parser.add_argument('--temperature-end', type=float, default=None,
                        help='Target temperature at the final iteration. Linear anneal from --temperature.')
    parser.add_argument('--modal-workers', type=int, default=0,
                        help='If >0, run self-play on Modal with this many workers instead of locally. '
                             'Each worker gets (self-play-games / modal-workers) games. '
                             'Requires Modal setup (pip install modal && python3 -m modal setup).')
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
        actual_iter = iteration + args.iter_offset
        print(f"{'='*60}")
        print(f"  ITERATION {iteration}/{args.iterations} (file iter={actual_iter})")
        print(f"{'='*60}")

        # 1. Generate self-play data (fresh each iteration)
        self_play_path = f"self_play_iter{actual_iter}.bin"
        if os.path.exists(self_play_path):
            os.remove(self_play_path)
        strategy = "NNUE" if current_weights else "greedy"

        # Compute current epsilon from annealing schedule (if --epsilon-end set)
        if args.epsilon_end is not None and args.iterations > 1:
            progress = (iteration - 1) / (args.iterations - 1)
            current_epsilon = args.epsilon - progress * (args.epsilon - args.epsilon_end)
        else:
            current_epsilon = args.epsilon
        print(f"  ε = {current_epsilon:.3f}")

        # Compute current temperature (for SA) if annealing
        current_temperature = None
        if args.temperature is not None:
            if args.temperature_end is not None and args.iterations > 1:
                progress = (iteration - 1) / (args.iterations - 1)
                current_temperature = args.temperature - progress * (args.temperature - args.temperature_end)
            else:
                current_temperature = args.temperature
            print(f"  T = {current_temperature:.3f} (softmax sampling)")

        if args.modal_workers > 0:
            run_self_play_modal(
                args.self_play_games,
                current_weights,
                self_play_path,
                epsilon=current_epsilon,
                aux_targets=args.aux_targets or args.split_value_head,
                temperature=current_temperature,
                num_workers=args.modal_workers,
            )
        else:
            run_self_play(
                args.self_play_games,
                current_weights,
                self_play_path,
                epsilon=current_epsilon,
                top_pct=args.top_pct,
                aux_targets=args.aux_targets or args.split_value_head,
                opp_weights=args.opp_weights,
                temperature=current_temperature,
            )

        # 2. Merge self-play + MCE expert data (fresh each iteration, no accumulation)
        # When using aux targets / split heads, the MCE cache must also be MCV3 so it
        # has aux + target_wildlife fields. The cache has been upgraded to MCV3 — mix it
        # in unless explicitly disabled.
        merged_path = f"training_merged_iter{actual_iter}.bin"
        if os.path.exists(merged_path):
            os.remove(merged_path)
        files_to_merge = [self_play_path]
        if not args.no_mce and args.mce_samples != 'none' and os.path.exists(args.mce_samples):
            # Check magic byte to ensure we only mix compatible formats
            with open(args.mce_samples, 'rb') as f:
                magic = f.read(4)
            if args.aux_targets or args.split_value_head:
                if magic != b'MCV3':
                    print(f"  Skipping MCE cache {args.mce_samples}: magic={magic!r}, "
                          f"need MCV3 for aux/split training")
                else:
                    files_to_merge.append(args.mce_samples)
            else:
                files_to_merge.append(args.mce_samples)
        merge_samples(files_to_merge, merged_path)

        # 3. Train
        iter_weights = f"{args.iter_prefix}{actual_iter}.bin"
        run_pytorch_training(
            merged_path,
            epochs=args.epochs_per_iter,
            lr=args.lr,
            init_weights=current_weights,
            out_weights=iter_weights,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            use_aux=args.aux_targets,
            aux_bear_weight=args.aux_bear_weight,
            aux_salmon_weight=args.aux_salmon_weight,
            split_value_head=args.split_value_head,
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
