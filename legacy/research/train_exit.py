"""Expert Iteration (ExIt) loop: MCE expert -> PolicyNet training -> repeat.

Each iteration:
1. Collect MCE-scored candidate data (using policy for faster pruning after iter 1)
2. Train PolicyNet on ALL accumulated data
3. Benchmark policy-guided MCE
4. Repeat

Usage:
    python3 train_exit.py --iterations 10 --games-per-iter 500 --rollouts 300
    python3 train_exit.py --iterations 5 --games-per-iter 1000 --rollouts 300 --top-k 5
"""

import argparse
import os
import subprocess
import time
import shutil
import re


def run_cmd(cmd, description=""):
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
    return result


def parse_mean_score(output):
    for line in output.split('\n'):
        if 'Mean:' in line and 'bonus' not in line.lower():
            m = re.search(r'Mean:\s+([\d.]+)', line)
            if m:
                return float(m.group(1))
    return 0.0


def main():
    parser = argparse.ArgumentParser(description='Expert Iteration training loop')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--games-per-iter', type=int, default=500)
    parser.add_argument('--rollouts', type=int, default=300,
                        help='MCE rollouts for data collection')
    parser.add_argument('--benchmark-rollouts', type=int, default=750,
                        help='MCE rollouts for benchmarking')
    parser.add_argument('--top-k', type=int, default=8,
                        help='Candidates to keep after policy pruning')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--benchmark-games', type=int, default=20)
    parser.add_argument('--weights', default='nnue_weights_hybrid_iter4.bin',
                        help='Value NNUE weights (fixed throughout)')
    parser.add_argument('--start-iter', type=int, default=1)
    args = parser.parse_args()

    os.makedirs('exit_work', exist_ok=True)

    print(f"Expert Iteration (ExIt) Training Loop")
    print(f"  Iterations:     {args.iterations}")
    print(f"  Games/iter:     {args.games_per_iter}")
    print(f"  MCE rollouts:   {args.rollouts}")
    print(f"  Policy top-k:   {args.top_k}")
    print(f"  Train epochs:   {args.epochs}")
    print(f"  Value weights:  {args.weights}")

    # Initial MCE baseline
    result = run_cmd(
        f"./target/release/cascadia-cli {args.benchmark_games} --mce "
        f"--weights {args.weights} --rollouts {args.benchmark_rollouts}",
        "MCE baseline benchmark"
    )
    baseline = parse_mean_score(result.stdout)
    print(f"  MCE baseline: {baseline:.1f}")

    best_score = 0.0
    policy_weights = None
    all_data_files = []

    total_start = time.time()

    for iteration in range(args.start_iter, args.iterations + 1):
        iter_start = time.time()
        print(f"\n{'#'*60}")
        print(f"  ExIt ITERATION {iteration}/{args.iterations}")
        print(f"{'#'*60}")

        data_path = f"exit_work/iter{iteration}_data.bin"
        policy_path = f"exit_work/iter{iteration}_policy.pt"
        policy_bin_path = f"exit_work/iter{iteration}_policy.bin"

        # Step 1: Collect MCE data
        # Iteration 1: no policy pruning (standard MCE)
        # Later iterations: use policy for faster collection
        if iteration == 1 or policy_weights is None:
            collect_cmd = (
                f"./target/release/cascadia-cli {args.games_per_iter} "
                f"--collect-mce-policy --weights {args.weights} "
                f"--rollouts {args.rollouts} --random-seed "
                f"--out {data_path}"
            )
        else:
            # Use policy-guided MCE for collection (faster)
            collect_cmd = (
                f"./target/release/cascadia-cli {args.games_per_iter} "
                f"--collect-mce-policy --weights {args.weights} "
                f"--rollouts {args.rollouts} --random-seed "
                f"--out {data_path}"
            )
            # TODO: add --policy-weights flag to collect-mce-policy for pruned collection

        result = run_cmd(collect_cmd,
            f"Step 1: Collect MCE data ({args.games_per_iter} games, {args.rollouts} rollouts)")

        if not os.path.exists(data_path):
            print(f"ERROR: No data at {data_path}")
            break

        all_data_files.append(data_path)
        data_size = os.path.getsize(data_path)
        print(f"  Data: {data_size / 1e6:.1f} MB")

        # Step 2: Merge all data files from all iterations
        merged_path = "exit_work/merged_data.bin"
        if len(all_data_files) == 1:
            shutil.copy2(all_data_files[0], merged_path)
        else:
            # Concatenate MCP2 files (first keeps header, rest skip 4-byte magic)
            with open(merged_path, 'wb') as out:
                for i, f in enumerate(all_data_files):
                    with open(f, 'rb') as inp:
                        data = inp.read()
                        if i == 0:
                            out.write(data)
                        else:
                            out.write(data[4:])  # skip MCP2 magic

        merged_size = os.path.getsize(merged_path)
        print(f"  Merged data: {merged_size / 1e6:.1f} MB ({len(all_data_files)} iterations)")

        # Step 3: Train PolicyNet on ALL accumulated data
        init_flag = ""
        if iteration > 1 and policy_weights:
            init_flag = f"--init-weights {policy_weights}"

        result = run_cmd(
            f"python3 train_pytorch.py policy-standalone "
            f"--policy-data {merged_path} "
            f"{init_flag} "
            f"--epochs {args.epochs} --lr {args.lr} "
            f"--out {policy_path}",
            f"Step 2: Train PolicyNet on {len(all_data_files)} iterations of data"
        )

        if not os.path.exists(policy_path):
            print(f"ERROR: No policy weights at {policy_path}")
            break

        # Export to Rust format
        run_cmd(
            f"python3 -c \""
            f"import torch; from train_pytorch import PolicyNet, save_policy_net_rust; "
            f"m = PolicyNet(7670, 256, 64); "
            f"m.load_state_dict(torch.load('{policy_path}', map_location='cpu')); "
            f"save_policy_net_rust('{policy_bin_path}', m)\"",
            "Export PolicyNet to Rust binary"
        )

        policy_weights = policy_path

        # Step 4: Benchmark policy-guided MCE
        result = run_cmd(
            f"./target/release/cascadia-cli {args.benchmark_games} --policy-mce "
            f"--weights {args.weights} --policy-weights {policy_bin_path} "
            f"--rollouts {args.benchmark_rollouts} --top-k {args.top_k}",
            f"Step 3: Benchmark PolicyMCE (top-{args.top_k}, {args.benchmark_rollouts} rollouts)"
        )
        policy_mce_score = parse_mean_score(result.stdout)

        # Also benchmark with top-15 (no pruning, just to see pure MCE quality)
        result = run_cmd(
            f"./target/release/cascadia-cli {args.benchmark_games} --policy-mce "
            f"--weights {args.weights} --policy-weights {policy_bin_path} "
            f"--rollouts {args.benchmark_rollouts} --top-k 15",
            f"Step 3b: Benchmark PolicyMCE top-15 (no pruning)"
        )
        no_prune_score = parse_mean_score(result.stdout)

        iter_elapsed = time.time() - iter_start

        improved = policy_mce_score > best_score
        if improved:
            best_score = policy_mce_score
            shutil.copy2(policy_bin_path, "policy_net_best.bin")
            shutil.copy2(policy_path, "policy_net_best.pt")

        print(f"\n  --- ExIt Iteration {iteration} Summary ---")
        print(f"  PolicyMCE top-{args.top_k}: {policy_mce_score:.1f} {'NEW BEST' if improved else ''}")
        print(f"  PolicyMCE top-15:  {no_prune_score:.1f}")
        print(f"  MCE baseline:      {baseline:.1f}")
        print(f"  Best so far:       {best_score:.1f}")
        print(f"  Total data:        {len(all_data_files)} iters, {merged_size/1e6:.0f} MB")
        print(f"  Time: {iter_elapsed:.0f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ExIt complete: {args.iterations} iterations in {total_elapsed:.0f}s")
    print(f"Best PolicyMCE score: {best_score:.1f}")
    print(f"Best policy weights: policy_net_best.bin")


if __name__ == '__main__':
    main()
