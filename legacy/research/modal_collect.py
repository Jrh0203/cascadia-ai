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

# Shared Volume for all-in-Modal training: self-play data, checkpoints, iter weights
training_volume = modal.Volume.from_name("cascadia-training", create_if_missing=True)

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
    .add_local_file("nnue_weights_hybrid_iter4.bin", remote_path="/app/nnue_weights_hybrid_iter4.bin")
    .add_local_file("nnue_weights_hybrid_iter20.bin", remote_path="/app/nnue_weights_hybrid_iter20.bin")
    .add_local_file("nnue_weights_v3_iter10.bin", remote_path="/app/nnue_weights_v3_iter10.bin")
    .add_local_file("nnue_weights_v3_iter17.bin", remote_path="/app/nnue_weights_v3_iter17.bin")
    .add_local_file("nnue_weights_v3_iter20.bin", remote_path="/app/nnue_weights_v3_iter20.bin")
    .add_local_file("nnue_weights_v9_iter14.bin", remote_path="/app/nnue_weights_v9_iter14.bin")
)

# GPU image for training: CUDA-enabled PyTorch + the Rust binary + train_pytorch.py
# CUDA 12.1 is broadly supported; torch 2.x works with cu121 wheels.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "gcc", "pkg-config", "libssl-dev")
    .pip_install(
        "torch==2.4.0",
        "numpy>=1.26",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
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
    .add_local_file("train_pytorch.py", remote_path="/app/train_pytorch.py")
)


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=7200,
)
def self_play_worker(
    num_games: int,
    seed_offset: int,
    epsilon: float = 0.1,
    weights_data: bytes = b"",
    aux_targets: bool = False,
    temperature: float = -1.0,
) -> bytes:
    """Generate self-play training data on Modal. Returns .bin bytes."""
    import time as _time
    _start = _time.time()

    weights_path = None
    if weights_data:
        weights_path = "/tmp/current_weights.bin"
        with open(weights_path, "wb") as f:
            f.write(weights_data)

    out_path = "/tmp/self_play.bin"
    cmd = [
        "/app/target/release/cascadia-cli",
        str(num_games),
        "--self-play",
        "--random-seed",
        "--out", out_path,
    ]
    if weights_path:
        cmd.extend(["--weights", weights_path])
    if temperature > 0:
        cmd.extend(["--temperature", str(temperature)])
    else:
        cmd.extend(["--epsilon", str(epsilon)])
    if aux_targets:
        cmd.append("--aux-targets")

    env = os.environ.copy()
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app", env=env)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Self-play failed: {result.stderr}")

    _elapsed = _time.time() - _start
    _cost = _elapsed * 0.000014 * 8
    print(f"Worker cost: ${_cost:.3f} ({_elapsed:.0f}s, {num_games} games)")

    with open(out_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def self_play(
    num_workers: int = 20,
    games_per_worker: int = 5000,
    epsilon: float = 0.1,
    weights: str = "",
    aux_targets: bool = False,
    temperature: float = -1.0,
    out: str = "self_play_modal.bin",
):
    """Generate self-play data across Modal workers.

    Usage:
        modal run modal_collect.py::self_play --num-workers 20 --games-per-worker 5000
        modal run modal_collect.py::self_play --weights nnue_weights_v3_iter20.bin --aux-targets
    """
    import time as _time
    _start = _time.time()
    total_games = num_workers * games_per_worker
    print(f"Modal self-play: {num_workers} workers × {games_per_worker} = {total_games} games")

    weights_data = b""
    if weights and os.path.exists(weights):
        with open(weights, "rb") as f:
            weights_data = f.read()
        print(f"Uploading weights: {weights} ({len(weights_data)/1e6:.1f} MB)")
    else:
        print("No weights — greedy self-play")

    futures = [
        self_play_worker.spawn(
            games_per_worker,
            seed_offset=i * games_per_worker,
            epsilon=epsilon,
            weights_data=weights_data,
            aux_targets=aux_targets,
            temperature=temperature,
        )
        for i in range(num_workers)
    ]

    total_bytes = 0
    with open(out, "wb") as f:
        for i, future in enumerate(futures):
            data = future.get()
            if i == 0:
                f.write(data)
            else:
                f.write(data[4:])  # skip magic header
            total_bytes += len(data)
            print(f"  Worker {i+1}/{num_workers}: {len(data)/1e6:.1f} MB")

    _elapsed = _time.time() - _start
    _cost = _elapsed * num_workers * 0.000014 * 8
    print(f"\nWrote {out} ({total_bytes/1e6:.1f} MB, {total_games} games)")
    print(f"Wall time: {_elapsed:.0f}s, est. cost: ${_cost:.2f}")


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=7200,
)
def collect_games(num_games: int, rollouts: int = 300) -> bytes:
    """Run MCE collection for num_games, return the samples file bytes."""
    import time as _time
    _start = _time.time()
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

    _elapsed = _time.time() - _start
    _cost = _elapsed * 0.000014 * 8  # ~$0.000014/sec/vCPU × 8 vCPUs
    print(f"Worker cost: ${_cost:.3f} ({_elapsed:.0f}s)")

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
def benchmark_custom(
    num_games: int,
    seed_offset: int,
    extra_args: list,          # e.g. ["--nnue-rollout-mce", "--rollouts", "200", "--alloc", "halving", "--candidates", "expanded", "--prefilter-k", "8"]
    weights: str = "nnue_weights_v9_iter14.bin",
    extra_env: dict = None,    # e.g. {"MCE_CV_ALPHA": "0.85", "MCE_LMR": "1"}
) -> str:
    """Run cascadia-cli with arbitrary extra args + env vars. Returns stdout."""
    import time as _time
    _start = _time.time()
    weights_path = f"/app/{weights}"
    cmd = ["/app/target/release/cascadia-cli", str(num_games)]
    cmd.extend(extra_args)
    cmd.extend(["--weights", weights_path])
    env = os.environ.copy()
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app", env=env)
    _elapsed = _time.time() - _start
    _cost = _elapsed * 0.000014 * 8
    print(f"Worker cost: ${_cost:.3f} ({_elapsed:.0f}s)")
    print(result.stderr)
    return result.stdout


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
    import time as _time
    _start = _time.time()
    weights_path = f"/app/{weights}"

    cmd = [
        "/app/target/release/cascadia-cli",
        str(num_games),
    ]

    if strategy == "mce":
        cmd.extend(["--mce", "--rollouts", str(rollouts)])
    elif strategy == "nnue":
        cmd.append("--nnue")
    elif strategy == "greedy_mce":
        cmd.extend(["--greedy-mce", "--rollouts", str(rollouts)])
    elif strategy == "greedy_mce_halving":
        cmd.extend(["--greedy-mce", "--rollouts", str(rollouts), "--alloc", "halving"])
    elif strategy == "nnue_rollout_mce":
        cmd.extend(["--nnue-rollout-mce", "--rollouts", str(rollouts), "--alloc", "halving"])
    else:
        pass  # greedy

    # greedy and greedy_mce don't need weights; others do.
    if strategy in ("mce", "nnue", "nnue_rollout_mce"):
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
    _elapsed = _time.time() - _start
    _cost = _elapsed * 0.000014 * 8
    print(f"Worker cost: ${_cost:.3f} ({_elapsed:.0f}s)")
    print(result.stderr)
    return result.stdout


@app.local_entrypoint()
def main():
    """Default entrypoint — show usage."""
    print("Usage:")
    print("  modal run modal_collect.py collect --num-workers 10 --games-per-worker 100")
    print("  modal run modal_collect.py benchmark --num-workers 10 --games-per-worker 50")


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=7200,
)
def collect_mce_selfplay(
    num_games: int,
    seed_offset: int,
    rollouts: int = 300,
    weights: str = "nnue_weights_hybrid_iter4.bin",
) -> tuple:
    """Play full MCE games, return value samples + policy samples as bytes."""
    import time as _time
    _start = _time.time()
    value_path = "/tmp/value_samples.bin"
    policy_path = "/tmp/policy_samples.bin"
    weights_path = f"/app/{weights}"

    env = os.environ.copy()
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)

    result = subprocess.run(
        [
            "/app/target/release/cascadia-cli",
            str(num_games),
            "--mce-selfplay",
            "--weights", weights_path,
            "--rollouts", str(rollouts),
            "--random-seed",
            "--out", value_path,
            "--policy-out", policy_path,
        ],
        capture_output=True,
        text=True,
        cwd="/app",
        env=env,
    )
    print(result.stderr)
    print(result.stdout)

    _elapsed = _time.time() - _start
    _cost = _elapsed * 0.000014 * 8
    print(f"Worker cost: ${_cost:.3f} ({_elapsed:.0f}s)")

    value_data = b""
    policy_data = b""
    if os.path.exists(value_path):
        with open(value_path, "rb") as f:
            value_data = f.read()
    if os.path.exists(policy_path):
        with open(policy_path, "rb") as f:
            policy_data = f.read()

    if not value_data:
        raise RuntimeError(f"No value output. stderr: {result.stderr}")

    return (value_data, policy_data)


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=7200,
)
def collect_exact_selfplay(
    num_games: int,
    seed_offset: int,
    weights: str = "nnue_weights_hybrid_iter4.bin",
) -> tuple:
    """Play full expectimax games, return value + policy samples as bytes."""
    import time as _time
    _start = _time.time()
    value_path = "/tmp/value_samples.bin"
    policy_path = "/tmp/policy_samples.bin"
    weights_path = f"/app/{weights}"

    env = os.environ.copy()
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)

    result = subprocess.run(
        [
            "/app/target/release/cascadia-cli",
            str(num_games),
            "--exact-selfplay",
            "--weights", weights_path,
            "--random-seed",
            "--out", value_path,
            "--policy-out", policy_path,
        ],
        capture_output=True,
        text=True,
        cwd="/app",
        env=env,
    )
    print(result.stderr)
    print(result.stdout)

    _elapsed = _time.time() - _start
    _cost = _elapsed * 0.000014 * 8
    print(f"Worker cost: ${_cost:.3f} ({_elapsed:.0f}s)")

    value_data = b""
    policy_data = b""
    if os.path.exists(value_path):
        with open(value_path, "rb") as f:
            value_data = f.read()
    if os.path.exists(policy_path):
        with open(policy_path, "rb") as f:
            policy_data = f.read()

    if not value_data:
        raise RuntimeError(f"No value output. stderr: {result.stderr}")

    return (value_data, policy_data)


@app.local_entrypoint()
def exact_selfplay(
    num_workers: int = 100,
    games_per_worker: int = 1000,
    weights: str = "nnue_weights_hybrid_iter4.bin",
):
    """Play expectimax games across workers, collect value + policy training data."""
    total_games = num_workers * games_per_worker
    print(f"Exact selfplay: {num_workers} workers, {games_per_worker} games each = {total_games} total")
    print(f"Weights: {weights}")

    futures = [
        collect_exact_selfplay.spawn(
            games_per_worker,
            seed_offset=i * games_per_worker,
            weights=weights,
        )
        for i in range(num_workers)
    ]

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    value_path = f"exact_value_{timestamp}.bin"
    policy_path = f"exact_policy_{timestamp}.bin"

    total_value_bytes = 0
    total_policy_bytes = 0
    with open(value_path, "wb") as vf, open(policy_path, "wb") as pf:
        for i, future in enumerate(futures):
            value_data, policy_data = future.get()
            if i == 0:
                vf.write(value_data)
                pf.write(policy_data)
            else:
                vf.write(value_data[4:])  # skip MCEP magic
                pf.write(policy_data[4:])  # skip MCP2 magic
            total_value_bytes += len(value_data)
            total_policy_bytes += len(policy_data)
            print(f"  Worker {i+1}/{num_workers} done: {len(value_data)} value, {len(policy_data)} policy")

    print(f"\nValue samples: {value_path} ({total_value_bytes/1e6:.1f} MB)")
    print(f"Policy samples: {policy_path} ({total_policy_bytes/1e6:.1f} MB)")
    print(f"Total games: {total_games}")


@app.local_entrypoint()
def mce_selfplay(
    num_workers: int = 15,
    games_per_worker: int = 200,
    rollouts: int = 300,
    weights: str = "nnue_weights_hybrid_iter4.bin",
):
    """Play MCE games across workers, collect value + policy training data."""
    total_games = num_workers * games_per_worker
    print(f"MCE self-play: {num_workers} workers, {games_per_worker} games each = {total_games} total")
    print(f"Rollouts: {rollouts}, weights: {weights}")

    futures = [
        collect_mce_selfplay.spawn(
            games_per_worker,
            seed_offset=i * games_per_worker,
            rollouts=rollouts,
            weights=weights,
        )
        for i in range(num_workers)
    ]

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    value_path = f"mce_value_{timestamp}.bin"
    policy_path = f"mce_policy_{timestamp}.bin"

    total_value_bytes = 0
    total_policy_bytes = 0
    with open(value_path, "wb") as vf, open(policy_path, "wb") as pf:
        for i, future in enumerate(futures):
            value_data, policy_data = future.get()
            if i == 0:
                vf.write(value_data)
                pf.write(policy_data)
            else:
                vf.write(value_data[4:])  # skip MCEP magic
                pf.write(policy_data[4:])  # skip MCP2 magic
            total_value_bytes += len(value_data)
            total_policy_bytes += len(policy_data)
            print(f"  Worker {i+1}/{num_workers} done: {len(value_data)} value bytes, {len(policy_data)} policy bytes")

    print(f"\nValue samples: {value_path} ({total_value_bytes/1e6:.1f} MB)")
    print(f"Policy samples: {policy_path} ({total_policy_bytes/1e6:.1f} MB)")
    print(f"Total games: {total_games}")


@app.local_entrypoint()
def collect(
    num_workers: int = 1,
    games_per_worker: int = 100,
    rollouts: int = 300,
):
    """Collect MCE training data across multiple workers."""
    import time as _time
    _total_start = _time.time()
    print(f"Collecting: {num_workers} workers, {games_per_worker} games each, rollouts={rollouts}")
    print(f"Total: {num_workers * games_per_worker} games")

    futures = [
        collect_games.spawn(games_per_worker, rollouts)
        for _ in range(num_workers)
    ]

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"mce_samples_{timestamp}.bin"
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

    _total_elapsed = _time.time() - _total_start
    _total_cost = _total_elapsed * num_workers * 0.000014 * 8
    print(f"\nWrote {output_path} ({total_bytes} bytes from {num_workers} workers)")
    print(f"Total games: {num_workers * games_per_worker}")
    print(f"Estimated cost: ${_total_cost:.2f} ({_total_elapsed:.0f}s wall time × {num_workers} workers)")
    print(f"\nTo merge with existing data:")
    print(f"  python3 -c \"old=open('mce_policy_samples.bin','rb').read(); f=open('{output_path}','ab'); f.write(old[4:])\"")
    print(f"  mv {output_path} mce_policy_samples.bin")


@app.local_entrypoint()
def validate_winners(
    num_workers: int = 10,
    games_per_worker: int = 20,
    weights: str = "nnue_weights_v9_iter14.bin",
):
    """N=200 validation of top candidates from bench_new_features.

    Variants:
      V0_baseline_200r    — expanded+pf8+halving @ 200 rollouts (current champion)
      V1_sr_200r          — same but --alloc sr (had bonus edge at N=50)
      V2_baseline_750r    — expanded+pf8+halving @ 750 rollouts (local N=30 showed 97.5)
    """
    base_args_200 = ["--nnue-rollout-mce", "--rollouts", "200",
                     "--alloc", "halving", "--candidates", "expanded",
                     "--prefilter-k", "8"]
    sr_args_200   = ["--nnue-rollout-mce", "--rollouts", "200",
                     "--alloc", "sr", "--candidates", "expanded",
                     "--prefilter-k", "8"]
    base_args_750 = ["--nnue-rollout-mce", "--rollouts", "750",
                     "--alloc", "halving", "--candidates", "expanded",
                     "--prefilter-k", "8"]

    variants = [
        ("V0_baseline_200r", base_args_200, {}, games_per_worker),
        ("V1_sr_200r",       sr_args_200,   {}, games_per_worker),
        ("V2_baseline_750r", base_args_750, {}, max(5, games_per_worker // 2)),
    ]

    all_futures = []
    for name, args, env_dict, gpw in variants:
        for i in range(num_workers):
            f = benchmark_custom.spawn(
                gpw,
                seed_offset=i * gpw,
                extra_args=args,
                weights=weights,
                extra_env=env_dict,
            )
            all_futures.append((name, f))

    print(f"Spawned {len(all_futures)} worker tasks")
    print()

    results_by_variant = {}
    for i, (name, fut) in enumerate(all_futures):
        output = fut.get()
        results_by_variant.setdefault(name, []).append(output)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(all_futures)} workers done")

    import re
    print("\n" + "=" * 78)
    print("═══ VALIDATION RESULTS ═══")
    print(f"{'Variant':<24} {'Base':>7} {'Bonus':>7} {'Workers':>8} {'N':>5}")
    print("-" * 78)
    for name, _, _, gpw in variants:
        outputs = results_by_variant.get(name, [])
        base_means = []
        bonus_means = []
        for out in outputs:
            mb = re.search(r"Base Score.*?Mean:\s*(\S+)", out, re.DOTALL)
            mB = re.search(r"With Habitat Bonus:\s*\n\s*Mean:\s*(\S+)", out)
            if mb: base_means.append(float(mb.group(1)))
            if mB: bonus_means.append(float(mB.group(1)))
        if base_means:
            base_pooled = sum(base_means) / len(base_means)
            bonus_pooled = sum(bonus_means) / len(bonus_means) if bonus_means else 0.0
            n_eff = len(base_means) * gpw
            print(f"{name:<24} {base_pooled:>7.2f} {bonus_pooled:>7.2f} {len(base_means):>8} {n_eff:>5}")
        else:
            print(f"{name:<24} {'?':>7} {'?':>7}")
    print("\nRaw per-worker means:")
    for name, _, _, _ in variants:
        outputs = results_by_variant.get(name, [])
        means = []
        for out in outputs:
            mb = re.search(r"Base Score.*?Mean:\s*(\S+)", out, re.DOTALL)
            if mb: means.append(float(mb.group(1)))
        if means:
            import statistics
            pm = sum(means) / len(means)
            try: sd = statistics.stdev(means)
            except statistics.StatisticsError: sd = 0.0
            se = sd / (len(means) ** 0.5)
            print(f"  {name}: means={[f'{m:.1f}' for m in means]} pooled={pm:.2f} SE={se:.3f}")


@app.local_entrypoint()
def bench_new_features(
    num_workers: int = 10,
    games_per_worker: int = 20,
    rollouts: int = 200,
    weights: str = "nnue_weights_v9_iter14.bin",
):
    """Run all 9 new-feature variants on Modal at scale.

    Total games per variant: num_workers * games_per_worker.
    Default: 10 workers × 20 games = N=200 per variant (SE ~0.35).

    Variants (all use --candidates expanded --prefilter-k 8 --alloc halving):
      30_baseline             — no env, baseline
      31_cv_0.85              — MCE_CV_ALPHA=0.85
      32_cv_0.70              — MCE_CV_ALPHA=0.70
      33_lmr                  — MCE_LMR=1
      34_strategy             — MCE_STRATEGY_BIAS=1
      35_cv_lmr_strat         — all three env vars
      36_sr                   — --alloc sr (successive rejects)
      37_pw                   — --alloc halving-pw (progressive widening)
      38_sr_cv_lmr_strat      — sr + all env vars

    Cost estimate: 9 variants × 10 workers × ~10 min @ 8 vCPU × $0.000014/vcpu-s
      ≈ 9 × 10 × 600 × 8 × 0.000014 ≈ $6.05
    """
    base_args = ["--nnue-rollout-mce", "--rollouts", str(rollouts),
                 "--alloc", "halving", "--candidates", "expanded",
                 "--prefilter-k", "8"]
    sr_args   = ["--nnue-rollout-mce", "--rollouts", str(rollouts),
                 "--alloc", "sr", "--candidates", "expanded",
                 "--prefilter-k", "8"]
    pw_args   = ["--nnue-rollout-mce", "--rollouts", str(rollouts),
                 "--alloc", "halving-pw", "--candidates", "expanded",
                 "--prefilter-k", "8"]

    variants = [
        ("30_baseline",        base_args, {}),
        ("31_cv_0.85",         base_args, {"MCE_CV_ALPHA": "0.85"}),
        ("32_cv_0.70",         base_args, {"MCE_CV_ALPHA": "0.70"}),
        ("33_lmr",             base_args, {"MCE_LMR": "1"}),
        ("34_strategy",        base_args, {"MCE_STRATEGY_BIAS": "1"}),
        ("35_cv_lmr_strat",    base_args, {"MCE_CV_ALPHA": "0.85", "MCE_LMR": "1", "MCE_STRATEGY_BIAS": "1"}),
        ("36_sr",              sr_args,   {}),
        ("37_pw",              pw_args,   {}),
        ("38_sr_cv_lmr_strat", sr_args,   {"MCE_CV_ALPHA": "0.85", "MCE_LMR": "1", "MCE_STRATEGY_BIAS": "1"}),
    ]

    total_games = num_workers * games_per_worker
    print(f"Benchmarking {len(variants)} new-feature variants × {total_games} games each")
    print(f"Weights: {weights}, rollouts: {rollouts}")
    print()

    # Spawn ALL variants ALL workers in parallel (up to Modal's limits)
    all_futures = []  # (variant_name, future)
    for name, args, env_dict in variants:
        for i in range(num_workers):
            f = benchmark_custom.spawn(
                games_per_worker,
                seed_offset=i * games_per_worker,  # same seeds across variants
                extra_args=args,
                weights=weights,
                extra_env=env_dict,
            )
            all_futures.append((name, f))

    print(f"Spawned {len(all_futures)} worker tasks across {len(variants)} variants")
    print()

    # Collect by variant
    results_by_variant = {}
    for i, (name, fut) in enumerate(all_futures):
        output = fut.get()
        results_by_variant.setdefault(name, []).append(output)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(all_futures)} workers done")

    # Aggregate: pool worker means per variant (each worker has games_per_worker games,
    # so pooling N workers gives effective N×games_per_worker sample).
    import re
    print("\n" + "=" * 78)
    print("═══ NEW FEATURES — MODAL RESULTS ═══")
    print(f"{'Variant':<28} {'Base':>7} {'Bonus':>7} {'Workers':>8} {'N':>5}")
    print("-" * 78)
    for name, _, _ in variants:
        outputs = results_by_variant.get(name, [])
        base_means = []
        bonus_means = []
        for out in outputs:
            mb = re.search(r"Base Score.*?Mean:\s*(\S+)", out, re.DOTALL)
            mB = re.search(r"With Habitat Bonus:\s*\n\s*Mean:\s*(\S+)", out)
            if mb: base_means.append(float(mb.group(1)))
            if mB: bonus_means.append(float(mB.group(1)))
        if base_means:
            base_pooled = sum(base_means) / len(base_means)
            bonus_pooled = sum(bonus_means) / len(bonus_means) if bonus_means else 0.0
            n_eff = len(base_means) * games_per_worker
            print(f"{name:<28} {base_pooled:>7.2f} {bonus_pooled:>7.2f} {len(base_means):>8} {n_eff:>5}")
        else:
            print(f"{name:<28} {'?':>7} {'?':>7}")

    print("\nRaw per-worker means (for variance inspection):")
    for name, _, _ in variants:
        outputs = results_by_variant.get(name, [])
        means = []
        for out in outputs:
            mb = re.search(r"Base Score.*?Mean:\s*(\S+)", out, re.DOTALL)
            if mb: means.append(float(mb.group(1)))
        if means:
            print(f"  {name}: {[f'{m:.1f}' for m in means]}")


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
        stats = {}
        section = None  # track which section we're in
        for line in output_text.split("\n"):
            stripped = line.strip()
            if "Base Score" in line:
                section = "base"
            elif "With Habitat Bonus" in line:
                section = "bonus"
            elif section == "base":
                if stripped.startswith("Mean:"):
                    val = re.search(r"[\d.]+", stripped)
                    if val:
                        stats["base_mean"] = float(val.group())
                elif stripped.startswith("Median:"):
                    val = re.search(r"\d+", stripped)
                    if val:
                        stats["base_median"] = int(val.group())
                elif stripped.startswith("P10:"):
                    val = re.search(r"\d+", stripped)
                    if val:
                        stats["base_p10"] = int(val.group())
                elif stripped.startswith("P90:"):
                    val = re.search(r"\d+", stripped)
                    if val:
                        stats["base_p90"] = int(val.group())
                elif stripped.startswith("Min/Max:"):
                    m = re.search(r"(\d+)/(\d+)", stripped)
                    if m:
                        stats["base_min"] = int(m.group(1))
                        stats["base_max"] = int(m.group(2))
            elif section == "bonus":
                if stripped.startswith("Mean:"):
                    m = re.search(r"([\d.]+) \(\+([\d.]+)", stripped)
                    if m:
                        stats["bonus_mean"] = float(m.group(1))
                        stats["avg_bonus"] = float(m.group(2))
                elif stripped.startswith("P10:"):
                    val = re.search(r"\d+", stripped)
                    if val:
                        stats["bonus_p10"] = int(val.group())
                elif stripped.startswith("P90:"):
                    val = re.search(r"\d+", stripped)
                    if val:
                        stats["bonus_p90"] = int(val.group())
        return stats

    # Aggregate across workers per strategy (weighted average of means)
    print("\n" + "=" * 70)
    print(f"{'STRATEGY COMPARISON':^70}")
    print("=" * 70)
    print(f"{'Strategy':<18} {'Games':>5} {'Base':>6} {'w/Bonus':>8} {'(+Hab)':>7} {'P10':>5} {'P90':>5} {'Min':>5} {'Max':>5}")
    print("-" * 70)

    strategy_order = list(dict.fromkeys(l for l, _ in futures))
    for label in strategy_order:
        outputs = results_by_strategy[label]
        all_stats = [parse_scores(o) for o in outputs]
        n_workers = len(all_stats)
        n_games = n_workers * games_per_worker

        valid = [s for s in all_stats if "base_mean" in s]
        if not valid:
            print(f"{label:<18} {'(no data)':>5}")
            continue

        avg_base = sum(s["base_mean"] for s in valid) / len(valid)
        avg_bonus = sum(s.get("bonus_mean", s["base_mean"]) for s in valid) / len(valid)
        avg_bonus_delta = sum(s.get("avg_bonus", 0) for s in valid) / len(valid)
        min_p10 = min(s.get("base_p10", 0) for s in valid)
        max_p90 = max(s.get("base_p90", 0) for s in valid)
        min_score = min(s.get("base_min", 0) for s in valid)
        max_score = max(s.get("base_max", 0) for s in valid)

        bonus_str = f"{avg_bonus:.1f}" if avg_bonus_delta > 0 else "-"
        delta_str = f"+{avg_bonus_delta:.1f}" if avg_bonus_delta > 0 else "-"
        print(f"{label:<18} {n_games:>5} {avg_base:>6.1f} {bonus_str:>8} {delta_str:>7} {min_p10:>5} {max_p90:>5} {min_score:>5} {max_score:>5}")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# ALL-IN-MODAL TRAINING PIPELINE
#
# Everything runs on Modal: self-play (CPU workers), training (GPU), bench.
# Data flows through a Modal Volume so there's no upload/download between
# pipeline stages. The orchestrator (local) just issues .remote() calls and
# downloads weights after each iter for local monitoring.
# ═══════════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    volumes={"/data": training_volume},
    timeout=7200,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=5.0),
)
def selfplay_to_volume(
    iter_num: int,
    num_games: int,
    seed_offset: int,
    epsilon: float = 0.1,
    weights_path: str = "",  # path WITHIN the volume, e.g. "iter3/weights.bin"
    aux_targets: bool = True,
    temperature: float = -1.0,
    worker_id: int = 0,
) -> int:
    """Generate self-play data and write to the Modal Volume.

    Writes to /data/iter{N}/selfplay_{worker_id}.bin. Returns byte count.
    """
    import time as _time
    _start = _time.time()

    iter_dir = f"/data/iter{iter_num:02d}"
    os.makedirs(iter_dir, exist_ok=True)

    out_path = f"{iter_dir}/selfplay_{worker_id:02d}.bin"

    cmd = [
        "/app/target/release/cascadia-cli",
        str(num_games),
        "--self-play",
        "--random-seed",
        "--out", out_path,
    ]
    if weights_path:
        full_weights = f"/data/{weights_path}" if not weights_path.startswith("/") else weights_path
        if os.path.exists(full_weights):
            cmd.extend(["--weights", full_weights])
    if temperature > 0:
        cmd.extend(["--temperature", str(temperature)])
    else:
        cmd.extend(["--epsilon", str(epsilon)])
    if aux_targets:
        cmd.append("--aux-targets")

    env = os.environ.copy()
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app", env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Self-play failed: {result.stderr[-500:]}")

    size = os.path.getsize(out_path)
    training_volume.commit()
    _elapsed = _time.time() - _start
    _cost = _elapsed * 0.000014 * 8
    print(f"[SP iter{iter_num} w{worker_id}] {size/1e6:.0f}MB, {_elapsed:.0f}s, ${_cost:.3f}")
    return size


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    volumes={"/data": training_volume},
    timeout=1800,
)
def merge_selfplay(iter_num: int, num_workers: int) -> int:
    """Concatenate per-worker self-play files into one training.bin. Returns size."""
    iter_dir = f"/data/iter{iter_num:02d}"
    out_path = f"{iter_dir}/training.bin"
    total_bytes = 0
    with open(out_path, "wb") as out:
        for w in range(num_workers):
            p = f"{iter_dir}/selfplay_{w:02d}.bin"
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                data = f.read()
            if total_bytes == 0:
                out.write(data)  # include magic from first
            else:
                out.write(data[4:])  # skip magic for subsequent
            total_bytes += len(data)
    training_volume.commit()
    print(f"[MERGE iter{iter_num}] {num_workers} workers → {total_bytes/1e6:.0f}MB training.bin")
    return total_bytes


@app.function(
    image=gpu_image,
    gpu="L4",
    cpu=4,
    memory=32768,
    volumes={"/data": training_volume},
    timeout=10800,  # 3 hours max per iter
    # GPU preemption is common on Modal — auto-retry with exponential backoff so a
    # preempted training job resumes on a fresh worker instead of killing the run.
    # When combined with per-epoch checkpoints below, the retry continues from the
    # latest saved epoch rather than starting over.
    retries=modal.Retries(
        max_retries=5, backoff_coefficient=2.0, initial_delay=10.0,
    ),
)
def train_on_gpu(
    iter_num: int,
    init_weights_path: str = "",  # path in volume, e.g. "iter3/weights.bin"
    epochs: int = 15,
    lr: float = 0.0001,
    batch_size: int = 4096,
    hidden1: int = 512,
    hidden2: int = 64,
    num_features: int = 45260,
) -> dict:
    """Train one iteration on the GPU. Reads training.bin from volume, writes weights
    to iter{N}/weights.bin + per-epoch checkpoints to iter{N}/weights_epoch_{E}.bin.

    On preemption/retry: the most-recent per-epoch checkpoint in the volume is used
    as the init, and remaining epochs are continued. If iter{N}/weights.bin already
    exists from a prior completed attempt, this call is a no-op.
    Returns {'epochs': [...], 'final_rmse': float, 'elapsed': float}.
    """
    import time as _time
    import glob as _glob
    import re as _re
    _start = _time.time()

    iter_dir = f"/data/iter{iter_num:02d}"
    training_path = f"{iter_dir}/training.bin"
    out_weights = f"{iter_dir}/weights.bin"
    os.makedirs(iter_dir, exist_ok=True)

    # Fast-path: if iter{N}/weights.bin already exists from a prior attempt, skip.
    if os.path.exists(out_weights) and os.path.getsize(out_weights) > 1024:
        print(f"[TRAIN iter{iter_num}] weights.bin already exists ({os.path.getsize(out_weights)/1e6:.1f}MB) — skipping training")
        return {"epochs": [], "final_rmse": -1.0, "elapsed": 0.0, "resumed": True}

    # Resume from latest per-epoch checkpoint if present.
    ep_ckpts = sorted(
        _glob.glob(f"{iter_dir}/weights_epoch_*.bin"),
        key=lambda p: int(_re.search(r"weights_epoch_(\d+)", p).group(1)),
    )
    resume_epoch = 0
    resume_path = init_weights_path
    if ep_ckpts:
        latest = ep_ckpts[-1]
        m = _re.search(r"weights_epoch_(\d+)", latest)
        if m:
            resume_epoch = int(m.group(1))
            resume_path = latest.replace("/data/", "", 1)  # relative to volume
            print(f"[TRAIN iter{iter_num}] resuming from epoch {resume_epoch} → {latest}")
    if resume_epoch >= epochs:
        # Last per-epoch checkpoint is already past target — promote it to weights.bin.
        import shutil as _shutil
        _shutil.copy(ep_ckpts[-1], out_weights)
        training_volume.commit()
        return {"epochs": [], "final_rmse": -1.0, "elapsed": 0.0, "resumed": True}
    remaining_epochs = epochs - resume_epoch

    cmd = [
        "python3", "-u", "/app/train_pytorch.py", "value",
        "--samples", training_path,
        "--epochs", str(remaining_epochs),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--hidden1", str(hidden1),
        "--hidden2", str(hidden2),
        "--num-features", str(num_features),
        "--optimizer", "sgd",
        "--out", out_weights,
        "--no-augment",
        # Save a checkpoint after every epoch so a preempted worker can resume.
        # The checkpoint naming encodes the global epoch number (accounts for resume).
        "--save-every-epoch", f"{iter_dir}/weights_epoch_{{e}}.bin",
        "--epoch-offset", str(resume_epoch),
    ]
    if resume_path:
        full_init = f"/data/{resume_path}" if not resume_path.startswith("/") else resume_path
        if os.path.exists(full_init):
            cmd.extend(["--init-weights", full_init])

    # Stream output so we can see progress
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")
    stdout = result.stdout
    print(stdout[-2000:])  # tail of output
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        raise RuntimeError(f"Training failed (exit {result.returncode})")

    # Parse epoch RMSEs from stdout
    import re
    epoch_rmses = []
    for line in stdout.splitlines():
        m = re.search(r"Epoch \d+/\d+: RMSE=([\d.]+)", line)
        if m:
            epoch_rmses.append(float(m.group(1)))

    # Copy per-epoch checkpoints if train_pytorch.py saves them (it doesn't today,
    # but we save the final one which is always available).
    # For now, just ensure the final weights are committed.
    training_volume.commit()

    elapsed = _time.time() - _start
    _cost = elapsed * (0.80 / 3600.0)  # L4 at $0.80/hr
    final_rmse = epoch_rmses[-1] if epoch_rmses else -1.0
    print(f"[TRAIN iter{iter_num}] {len(epoch_rmses)} epochs, final RMSE={final_rmse:.3f}, {elapsed:.0f}s, ${_cost:.2f}")
    return {"epochs": epoch_rmses, "final_rmse": final_rmse, "elapsed": elapsed}


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    volumes={"/data": training_volume},
    timeout=600,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=5.0),
)
def bench_from_volume(weights_path: str, num_games: int = 50, strategy: str = "nnue", rollouts: int = 50) -> str:
    """Bench a weights file stored in the Modal Volume. Returns CLI stdout."""
    import time as _time
    _start = _time.time()

    full_weights = f"/data/{weights_path}" if not weights_path.startswith("/") else weights_path
    if not os.path.exists(full_weights):
        raise RuntimeError(f"Weights not found in volume: {weights_path}")

    cmd = ["/app/target/release/cascadia-cli", str(num_games)]
    if strategy == "mce":
        cmd.extend(["--mce", "--rollouts", str(rollouts)])
    else:
        cmd.append("--nnue")
    cmd.extend(["--weights", full_weights])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")
    elapsed = _time.time() - _start
    print(f"[BENCH {weights_path}] {elapsed:.0f}s")
    return result.stdout


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    volumes={"/data": training_volume},
    timeout=600,
)
def read_volume_file(path: str) -> bytes:
    """Read a file from the Modal Volume and return its bytes.
    Use this to download weights or any checkpoint to local disk."""
    full_path = f"/data/{path}" if not path.startswith("/") else path
    with open(full_path, "rb") as f:
        return f.read()


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    volumes={"/data": training_volume},
    timeout=120,
)
def list_volume_files(prefix: str = "") -> list:
    """List files under /data/{prefix}/. Returns list of (path, size) tuples."""
    import os
    base = f"/data/{prefix}" if prefix else "/data"
    out = []
    for root, dirs, files in os.walk(base):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, "/data")
            try:
                out.append((rel, os.path.getsize(p)))
            except OSError:
                pass
    return sorted(out)


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    volumes={"/data": training_volume},
    timeout=120,
)
def upload_file_to_volume(path: str, data: bytes) -> int:
    """Write raw bytes to /data/{path}. Returns size written."""
    full_path = f"/data/{path}"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as f:
        f.write(data)
    training_volume.commit()
    return len(data)


@app.local_entrypoint()
def list_checkpoints(prefix: str = ""):
    """List all files in the training volume under an optional prefix."""
    files = list_volume_files.remote(prefix)
    total = 0
    for path, size in files:
        print(f"  {size/1e6:>8.1f} MB  {path}")
        total += size
    print(f"  {'─'*70}")
    print(f"  {total/1e6:>8.1f} MB  total ({len(files)} files)")


@app.local_entrypoint()
def download_checkpoint(path: str, out: str = ""):
    """Download any file from the Modal Volume to local disk.

    Usage:
        modal run modal_collect.py::download_checkpoint --path v8/iter03/weights.bin
        modal run modal_collect.py::download_checkpoint --path v8/iter03/weights.bin --out local_weights.bin
    """
    if not out:
        out = os.path.basename(path)
    data = read_volume_file.remote(path)
    with open(out, "wb") as f:
        f.write(data)
    print(f"Downloaded {len(data)/1e6:.1f} MB → {out}")


@app.local_entrypoint()
def train_modal(
    run_name: str = "v8_modal",
    iterations: int = 15,
    self_play_games: int = 100000,
    num_workers: int = 20,
    epochs_per_iter: int = 15,
    lr: float = 0.0001,
    batch_size: int = 4096,
    hidden1: int = 512,
    hidden2: int = 64,
    num_features: int = 45260,
    epsilon: float = 0.1,
    init_weights: str = "",  # local file to upload as starting weights, or empty
    iter_offset: int = 0,    # continue from iter_offset+1
    benchmark_games: int = 50,
    download_every_iter: bool = True,  # auto-download weights after each iter
):
    """End-to-end training on Modal: self-play + GPU training + bench per iter.

    Each iter's final weights are downloaded to local disk as
    `{run_name}_iter{N}.bin`. Per-epoch checkpoints live in the Volume and can
    be pulled with `modal run modal_collect.py::download_checkpoint`.

    Usage:
        modal run modal_collect.py::train_modal --run-name v8 --iterations 15
    """
    import time as _time
    _total_start = _time.time()

    # Compute games per worker
    games_per_worker = self_play_games // num_workers

    # Upload initial weights to volume if provided
    best_weights_in_vol = ""
    if init_weights and os.path.exists(init_weights):
        with open(init_weights, "rb") as f:
            data = f.read()
        volume_path = f"{run_name}/init_weights.bin"
        upload_file_to_volume.remote(volume_path, data)
        best_weights_in_vol = volume_path
        print(f"Uploaded {init_weights} → volume:{volume_path} ({len(data)/1e6:.1f} MB)")

    print(f"═══ train_modal: {run_name} ═══")
    print(f"  Iterations:        {iterations} (starting at iter {iter_offset + 1})")
    print(f"  Self-play:         {num_workers} × {games_per_worker} = {self_play_games} games/iter")
    print(f"  Epochs/iter:       {epochs_per_iter}")
    print(f"  Architecture:      {num_features} → {hidden1} → {hidden2}")
    print(f"  LR:                {lr}")
    print(f"  GPU:               L4")
    print()

    for step in range(1, iterations + 1):
        iter_num = step + iter_offset
        print(f"┏━━ ITER {iter_num} (step {step}/{iterations}) ━━━━━━━━━━━━━━━━━━━")
        iter_start = _time.time()

        # 1. Self-play (parallel workers)
        sp_start = _time.time()
        print(f"┃ [{iter_num}] Self-play: spawning {num_workers} workers")
        seed = int(_time.time() * 1000) % (2**32)
        sp_futures = [
            selfplay_to_volume.spawn(
                iter_num=iter_num,
                num_games=games_per_worker,
                seed_offset=seed + w * games_per_worker,
                epsilon=epsilon,
                weights_path=best_weights_in_vol,
                aux_targets=True,
                temperature=-1.0,
                worker_id=w,
            )
            for w in range(num_workers)
        ]
        total_bytes = 0
        for f in sp_futures:
            total_bytes += f.get()
        sp_elapsed = _time.time() - sp_start
        print(f"┃ [{iter_num}] Self-play: {total_bytes/1e9:.1f}GB in {sp_elapsed:.0f}s")

        # 2. Merge self-play files
        merge_size = merge_selfplay.remote(iter_num=iter_num, num_workers=num_workers)

        # 3. Train on GPU
        # Path convention: {run_name}/iter{N}/weights.bin and training.bin
        # But selfplay_to_volume writes to iter{N}/training.bin without run prefix
        # so we namespace by run_name at a higher level. Keep simple: use iter{N}
        # subdirectory per run. If multiple runs need isolation, use different volumes.
        train_start = _time.time()
        print(f"┃ [{iter_num}] Training on L4 GPU ({epochs_per_iter} epochs)")
        init_path = best_weights_in_vol
        # Save final weights as iter{N}/weights.bin (shared-volume path)
        train_result = train_on_gpu.remote(
            iter_num=iter_num,
            init_weights_path=init_path,
            epochs=epochs_per_iter,
            lr=lr,
            batch_size=batch_size,
            hidden1=hidden1,
            hidden2=hidden2,
            num_features=num_features,
        )
        train_elapsed = _time.time() - train_start
        print(f"┃ [{iter_num}] Training done: final RMSE={train_result['final_rmse']:.3f} in {train_elapsed:.0f}s")

        # 4. Download weights to local disk
        weights_vol_path = f"iter{iter_num:02d}/weights.bin"
        if download_every_iter:
            dl_start = _time.time()
            data = read_volume_file.remote(weights_vol_path)
            local_path = f"nnue_weights_{run_name}_iter{iter_num}.bin"
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"┃ [{iter_num}] Downloaded weights → {local_path} ({len(data)/1e6:.1f} MB, {_time.time()-dl_start:.0f}s)")

        # 5. Benchmark
        bench_start = _time.time()
        bench_stdout = bench_from_volume.remote(
            weights_path=weights_vol_path,
            num_games=benchmark_games,
            strategy="nnue",
        )
        # Extract mean
        import re as _re
        m = _re.search(r"Mean:\s+([\d.]+)", bench_stdout)
        bench_mean = float(m.group(1)) if m else -1.0
        m2 = _re.search(r"With Habitat Bonus:\s*\n\s+Mean:\s+([\d.]+)", bench_stdout)
        bench_bonus = float(m2.group(1)) if m2 else -1.0
        bench_elapsed = _time.time() - bench_start
        print(f"┃ [{iter_num}] Bench: base={bench_mean:.1f}, w/bonus={bench_bonus:.1f} ({bench_elapsed:.0f}s)")

        # 6. Update best weights (this iter's output drives next iter)
        best_weights_in_vol = weights_vol_path

        iter_total = _time.time() - iter_start
        print(f"┗━━ ITER {iter_num} complete in {iter_total:.0f}s (SP:{sp_elapsed:.0f}s Train:{train_elapsed:.0f}s Bench:{bench_elapsed:.0f}s)")
        print()

    total_elapsed = _time.time() - _total_start
    print(f"═══ ALL DONE: {iterations} iters in {total_elapsed/60:.1f} min ═══")
    print(f"  Final weights: nnue_weights_{run_name}_iter{iter_offset + iterations}.bin")
