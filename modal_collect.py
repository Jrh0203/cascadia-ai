"""Modal app for distributed MCE sample collection and benchmarking.

Usage:
    # Collect training data
    modal run modal_collect.py collect --num-workers 10 --games-per-worker 100

    # Benchmark NNUE (fast, no rollouts)
    modal run modal_collect.py benchmark --num-workers 10 --games-per-worker 50 --strategy nnue

    # Benchmark MCE (with rollouts)
    modal run modal_collect.py benchmark --num-workers 10 --games-per-worker 10 --strategy mce --rollouts 750

    # Benchmark with custom weights (place .bin file in project root)
    modal run modal_collect.py benchmark --num-workers 10 --games-per-worker 50 --weights nnue_weights_v2.bin
"""

import modal
import subprocess
import os
import json

app = modal.App("cascadia")

# Build image: compile the Rust binary inside the container
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "gcc", "pkg-config", "libssl-dev")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .add_local_dir("crates", remote_path="/app/crates", copy=True)
    .add_local_file("Cargo.toml", remote_path="/app/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", remote_path="/app/Cargo.lock", copy=True)
    .run_commands(
        "cd /app && cargo build --release --bin cascadia-cli",
    )
    .add_local_file("nnue_weights_mce93.bin", remote_path="/app/nnue_weights_mce93.bin")
)


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=7200,
)
def collect_games(num_games: int, rollouts: int = 300) -> bytes:
    """Run MCE collection for num_games, return the samples file bytes."""
    out_path = "/tmp/samples.bin"
    result = subprocess.run(
        [
            "/app/target/release/cascadia-cli",
            str(num_games),
            "--collect-mce",
            "--weights", "/app/nnue_weights_mce93.bin",
            "--rollouts", str(rollouts),
            "--random-seed",
            "--out", out_path,
        ],
        capture_output=True,
        text=True,
        cwd="/app",
    )
    print(result.stderr)
    print(result.stdout)

    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            return f.read()
    else:
        raise RuntimeError(f"No output file. stderr: {result.stderr}")


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=7200,
)
def benchmark_games(
    num_games: int,
    seed_offset: int,
    strategy: str = "nnue",
    rollouts: int = 750,
    weights: str = "nnue_weights_mce93.bin",
) -> str:
    """Run benchmark for num_games starting at seed_offset. Returns stdout."""
    weights_path = f"/app/{weights}"

    cmd = [
        "/app/target/release/cascadia-cli",
        str(num_games),
    ]

    if strategy == "mce":
        cmd.extend(["--mce", "--rollouts", str(rollouts)])
    elif strategy == "nnue":
        cmd.append("--nnue")
    else:
        pass  # greedy

    if strategy in ("mce", "nnue"):
        cmd.extend(["--weights", weights_path])

    # Use deterministic seeds offset by worker index
    # The CLI uses seed_from_u64(i) for game i, so we shift game indices
    # by passing a higher num_games and... actually we need a seed offset flag.
    # For now, use env var to communicate the offset.
    env = os.environ.copy()
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/app",
        env=env,
    )
    print(result.stderr)
    return result.stdout


@app.local_entrypoint()
def main():
    """Default entrypoint — show usage."""
    print("Usage:")
    print("  modal run modal_collect.py collect --num-workers 10 --games-per-worker 100")
    print("  modal run modal_collect.py benchmark --num-workers 10 --games-per-worker 50")


@app.local_entrypoint()
def collect(
    num_workers: int = 1,
    games_per_worker: int = 100,
    rollouts: int = 300,
):
    """Collect MCE training data across multiple workers."""
    print(f"Collecting: {num_workers} workers, {games_per_worker} games each, rollouts={rollouts}")
    print(f"Total: {num_workers * games_per_worker} games")

    futures = [
        collect_games.spawn(games_per_worker, rollouts)
        for _ in range(num_workers)
    ]

    output_path = "mce_policy_samples.bin"
    total_bytes = 0
    with open(output_path, "wb") as out:
        for i, future in enumerate(futures):
            data = future.get()
            if i == 0:
                out.write(data)
            else:
                out.write(data[4:])  # skip magic header
            total_bytes += len(data)
            print(f"  Worker {i+1}/{num_workers} done: {len(data)} bytes")

    print(f"\nMerged {total_bytes} bytes from {num_workers} workers -> {output_path}")
    print(f"Total games: {num_workers * games_per_worker}")


@app.local_entrypoint()
def benchmark(
    num_workers: int = 1,
    games_per_worker: int = 50,
    strategy: str = "nnue",
    rollouts: int = 750,
    weights: str = "nnue_weights_mce93.bin",
):
    """Run benchmarks across multiple workers and aggregate results."""
    total_games = num_workers * games_per_worker
    print(f"Benchmarking: {num_workers} workers, {games_per_worker} games each")
    print(f"Strategy: {strategy}, rollouts: {rollouts}, weights: {weights}")
    print(f"Total: {total_games} games")

    futures = [
        benchmark_games.spawn(
            games_per_worker,
            seed_offset=i * games_per_worker,
            strategy=strategy,
            rollouts=rollouts,
            weights=weights,
        )
        for i in range(num_workers)
    ]

    all_output = []
    for i, future in enumerate(futures):
        output = future.get()
        all_output.append(output)
        print(f"  Worker {i+1}/{num_workers} done")

    # Print all worker outputs
    print("\n" + "=" * 60)
    for i, output in enumerate(all_output):
        print(f"\n--- Worker {i+1} ---")
        print(output)


@app.local_entrypoint()
def compare(
    strategies: str = "mce:300,mce:1500",
    workers_per_strategy: int = 3,
    games_per_worker: int = 10,
    weights: str = "nnue_weights_mce93.bin",
):
    """Compare multiple strategies head-to-head.

    All workers launch simultaneously across all strategies.
    Uses the same game seeds for fair comparison.

    Usage:
        modal run modal_collect.py compare --strategies "mce:300,mce:1500" --workers-per-strategy 3 --games-per-worker 10
        modal run modal_collect.py compare --strategies "nnue,mce:750" --workers-per-strategy 5 --games-per-worker 50
        modal run modal_collect.py compare --strategies "greedy,nnue,mce:300,mce:750"

    Format: strategy_name or strategy_name:rollouts (e.g., mce:300, nnue, greedy)
    """
    # Parse strategies
    strat_configs = []
    for s in strategies.split(","):
        s = s.strip()
        if ":" in s:
            name, rollouts = s.split(":", 1)
            strat_configs.append((name, int(rollouts)))
        else:
            strat_configs.append((s, 750))  # default rollouts

    total_workers = len(strat_configs) * workers_per_strategy
    total_games = total_workers * games_per_worker

    print(f"Comparing {len(strat_configs)} strategies, {workers_per_strategy} workers each, {games_per_worker} games/worker")
    print(f"Total: {total_workers} workers, {total_games} games")
    for name, rollouts in strat_configs:
        label = f"{name}({rollouts})" if name == "mce" else name
        print(f"  - {label}: {workers_per_strategy} workers × {games_per_worker} games = {workers_per_strategy * games_per_worker} games")

    # Launch ALL workers across all strategies simultaneously
    futures = []  # (strategy_label, future)
    for strat_name, rollouts in strat_configs:
        label = f"{strat_name}({rollouts})" if strat_name == "mce" else strat_name
        for i in range(workers_per_strategy):
            f = benchmark_games.spawn(
                games_per_worker,
                seed_offset=i * games_per_worker,  # same seeds across strategies for fair comparison
                strategy=strat_name,
                rollouts=rollouts,
                weights=weights,
            )
            futures.append((label, f))

    # Collect results grouped by strategy
    results_by_strategy = {}
    for label, future in futures:
        output = future.get()
        if label not in results_by_strategy:
            results_by_strategy[label] = []
        results_by_strategy[label].append(output)
        done = sum(len(v) for v in results_by_strategy.values())
        print(f"  {done}/{total_workers} workers done ({label})")

    # Parse scores from worker outputs and aggregate
    def parse_scores(output_text):
        """Extract base and bonus scores from CLI benchmark output."""
        import re
        base_scores = []
        bonus_scores = []
        # Look for score distribution lines like "  90- 94:     8 ████"
        # Or parse the Min/Max line for the range
        # Simplest: look for individual game scores in the output
        # Actually the CLI prints aggregate stats, not individual scores.
        # Parse the summary stats instead.
        stats = {}
        for line in output_text.split("\n"):
            line = line.strip()
            if line.startswith("Mean:"):
                val = re.search(r"[\d.]+", line)
                if val and "base" not in stats:
                    stats["base_mean"] = float(val.group())
            elif line.startswith("Median:"):
                val = re.search(r"\d+", line)
                if val and "base_median" not in stats:
                    stats["base_median"] = int(val.group())
            elif line.startswith("P10:"):
                val = re.search(r"\d+", line)
                if val and "base_p10" not in stats:
                    stats["base_p10"] = int(val.group())
            elif line.startswith("P90:"):
                val = re.search(r"\d+", line)
                if val and "base_p90" not in stats:
                    stats["base_p90"] = int(val.group())
            elif line.startswith("Min/Max:"):
                m = re.search(r"(\d+)/(\d+)", line)
                if m:
                    stats["base_min"] = int(m.group(1))
                    stats["base_max"] = int(m.group(2))
            elif "avg bonus" in line:
                m = re.search(r"([\d.]+) \(\+([\d.]+)", line)
                if m:
                    stats["bonus_mean"] = float(m.group(1))
                    stats["avg_bonus"] = float(m.group(2))
        return stats

    # Aggregate across workers per strategy (weighted average of means)
    print("\n" + "=" * 70)
    print(f"{'STRATEGY COMPARISON':^70}")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Games':>6} {'Base Mean':>10} {'+ Bonus':>10} {'P10':>6} {'P90':>6} {'Min':>6} {'Max':>6}")
    print("-" * 70)

    strategy_order = list(dict.fromkeys(l for l, _ in futures))
    for label in strategy_order:
        outputs = results_by_strategy[label]
        all_stats = [parse_scores(o) for o in outputs]
        n_workers = len(all_stats)
        n_games = n_workers * games_per_worker

        # Aggregate: average the means, min of mins, max of maxes, etc.
        valid = [s for s in all_stats if "base_mean" in s]
        if not valid:
            print(f"{label:<20} {'(no data)':>6}")
            continue

        avg_base = sum(s["base_mean"] for s in valid) / len(valid)
        avg_bonus = sum(s.get("bonus_mean", s["base_mean"]) for s in valid) / len(valid)
        avg_bonus_delta = sum(s.get("avg_bonus", 0) for s in valid) / len(valid)
        min_p10 = min(s.get("base_p10", 0) for s in valid)
        max_p90 = max(s.get("base_p90", 0) for s in valid)
        min_score = min(s.get("base_min", 0) for s in valid)
        max_score = max(s.get("base_max", 0) for s in valid)

        bonus_str = f"{avg_bonus:.1f} (+{avg_bonus_delta:.1f})" if avg_bonus_delta > 0 else "n/a"
        print(f"{label:<20} {n_games:>6} {avg_base:>10.1f} {bonus_str:>10} {min_p10:>6} {max_p90:>6} {min_score:>6} {max_score:>6}")

    print("=" * 70)
    print(f"(P10/P90 show worst-worker P10 and best-worker P90 across workers)")
