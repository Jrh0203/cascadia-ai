"""Modal: collect policy training data (features + MCE scores) in parallel.

Each worker runs 2 games of --train-policy --epochs 0, saves binary data,
returns as bytes. Local entrypoint aggregates into one file.

Usage:
    python3 -m modal run overnight/collect_policy_data_modal.py \
        --num-workers 50 --games-per-worker 2 \
        --weights nnue_weights_mid_fsp_iter10.bin
"""

import hashlib
import modal
import os
import sys
import time

app = modal.App("cascadia-policy-collect")

weights_volume = modal.Volume.from_name("cascadia-weights-cache", create_if_missing=True)

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
        "cd /app && cargo build --release --features mid-features --bin cascadia-cli",
    )
)


@app.function(image=image, cpu=8, memory=4096, timeout=1800,
              volumes={"/weights": weights_volume})
def collect_games(
    seed_offset: int,
    n_games: int,
    weight_name: str,
    rollouts_per_cand: int,
    max_candidates: int,
) -> bytes:
    """Collect policy training data for n_games. Returns binary PDAT bytes."""
    import subprocess as sp

    weights_path = f"/weights/{weight_name}"
    out_path = "/tmp/policy_data.bin"

    cmd = [
        "/app/target/release/cascadia-cli", str(n_games),
        "--train-policy",
        "--weights", "/tmp/dummy_out.bin",
        "--init-weights", weights_path,
        "--rollouts-per-cand", str(rollouts_per_cand),
        "--max-candidates", str(max_candidates),
        "--epochs", "0",
        "--save-policy-data", out_path,
    ]
    env = os.environ.copy()
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)
    env["MCE_MAX_EXTRA_CANDS"] = "150"

    result = sp.run(cmd, capture_output=True, text=True, cwd="/app", env=env)
    if result.returncode != 0:
        print(f"Worker stderr: {result.stderr[-500:]}")
        return b""

    print(result.stderr.strip())
    with open(out_path, "rb") as f:
        return f.read()


def _content_addressed_name(local_path: str) -> str:
    h = hashlib.sha256()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{os.path.splitext(os.path.basename(local_path))[0]}-{h.hexdigest()[:8]}.bin"


@app.local_entrypoint()
def run(
    num_workers: int = 50,
    games_per_worker: int = 2,
    weights: str = "nnue_weights_mid_fsp_iter10.bin",
    rollouts_per_cand: int = 100,
    max_candidates: int = 100,
    output: str = "overnight/policy_training_data.bin",
):
    total_games = num_workers * games_per_worker

    # Upload weights
    remote_name = _content_addressed_name(weights)
    print(f"Syncing weights...")
    try:
        existing = {e.path.lstrip("/") for e in weights_volume.iterdir("/")}
    except Exception:
        existing = set()
    if remote_name not in existing:
        with weights_volume.batch_upload() as batch:
            batch.put_file(weights, f"/{remote_name}")
        print(f"  Uploaded {weights}")
    else:
        print(f"  Already on volume")

    print(f"\nPolicy data collection:")
    print(f"  {total_games} games ({num_workers} workers × {games_per_worker} games)")
    print(f"  {max_candidates} candidates × {rollouts_per_cand} rollouts")
    print()

    t0 = time.time()
    futures = []
    for i in range(num_workers):
        f = collect_games.spawn(
            seed_offset=i * games_per_worker * 1000,
            n_games=games_per_worker,
            weight_name=remote_name,
            rollouts_per_cand=rollouts_per_cand,
            max_candidates=max_candidates,
        )
        futures.append((i, f))

    print(f"Dispatched {len(futures)} workers in {time.time() - t0:.1f}s")

    # Gather and concatenate binary data
    all_data = []
    total_positions = 0
    for i, f in futures:
        data = f.get()
        elapsed = time.time() - t0
        if data and len(data) > 8:
            # Parse header to count positions
            import struct
            n_pos = struct.unpack_from('<I', data, 4)[0]
            total_positions += n_pos
            all_data.append(data)
            print(f"  Worker {i}: {n_pos} positions, {len(data)} bytes ({elapsed:.0f}s)")
        else:
            print(f"  Worker {i}: EMPTY ({elapsed:.0f}s)")

    # Merge: write one combined PDAT file
    import struct
    with open(output, "wb") as out:
        out.write(b"PDAT")
        out.write(struct.pack('<I', total_positions))
        for data in all_data:
            # Skip each chunk's header (4 bytes magic + 4 bytes n_pos)
            out.write(data[8:])

    elapsed = time.time() - t0
    file_size = os.path.getsize(output)
    cost = num_workers * 300 * 8 * 0.000014  # ~300s per worker estimate
    print(f"\nDone: {total_positions} positions saved to {output} ({file_size/1e6:.1f} MB)")
    print(f"Wall clock: {elapsed:.0f}s  Estimated cost: ~${cost:.2f}")
