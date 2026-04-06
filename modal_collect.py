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
