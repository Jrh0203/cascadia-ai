"""Parallel FSP self-play data generation on Modal.

Each worker runs `cascadia-cli --selfplay-pool` with a fraction of the total
games, producing an MCV3-format shard. Local entrypoint concatenates shards
into a single training cache (stripping all but the first MCV3 magic header).

Usage:
    python3 -m modal run overnight/selfplay_fsp_modal.py \
        --total-games 50000 --num-workers 50 \
        --weights nnue_weights_v4opp_fsp_iter3.bin \
        --opp-pool "random,scarcity,preference,nnue_weights_mce93.bin,nnue_weights_mid_fsp_iter10.bin,nnue_weights_v4opp_fsp_iter3.bin" \
        --out /tmp/selfplay_fsp_shard.bin \
        --epsilon 0.1 \
        --seed-base 4217

Writes combined samples to --out (ready for `--cache-train`).
Cost estimate: ~$0.20-0.50 per 50K-game iteration.
"""
import hashlib
import modal
import os
import re
import sys
import time

app = modal.App("cascadia-selfplay-fsp")

weights_volume = modal.Volume.from_name("cascadia-weights-cache", create_if_missing=True)

# Build with mid-features + v4-opp so v4opp weights + older weights both load.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "gcc", "pkg-config", "libssl-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .add_local_dir("crates", remote_path="/app/crates", copy=True)
    .add_local_file("Cargo.toml", remote_path="/app/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", remote_path="/app/Cargo.lock", copy=True)
    .run_commands(
        "cd /app && cargo build --release --features mid-features,v4-opp --bin cascadia-cli",
    )
)


def _content_addressed_name(local_path: str) -> str:
    h = hashlib.sha256()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()[:8]
    base = os.path.basename(local_path)
    stem, ext = os.path.splitext(base)
    return f"{stem}-{digest}{ext}"


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=1800,
    volumes={"/weights": weights_volume},
)
def selfplay_worker(
    games: int,
    init_weight_name: str,     # basename on /weights/
    opp_pool_names: list,       # entries: "greedy", "random", "scarcity", "preference",
                                # or basenames on /weights/ (.bin files)
    seed: int,
    epsilon: float = 0.1,
    temperature: float = 0.0,
    player_mce: int = 0,        # 0 = NNUE-argmax (default); >0 = MCE(N) for player 0
) -> bytes:
    import subprocess as sp

    init_path = f"/weights/{init_weight_name}"

    # Translate opp-pool entries: tags pass through; .bin names prepend /weights/.
    pool_tokens = []
    for entry in opp_pool_names:
        if entry.lower() in ("greedy", "random", "scarcity", "preference"):
            pool_tokens.append(entry)
        else:
            pool_tokens.append(f"/weights/{entry}")
    pool_env = ",".join(pool_tokens)

    out_path = f"/tmp/shard_{seed}.bin"

    env = os.environ.copy()
    env["CASCADIA_TRAIN_OPP_POOL"] = pool_env
    env["CASCADIA_TRAIN_SEED"] = str(seed)
    if player_mce > 0:
        env["CASCADIA_TRAIN_PLAYER_MCE"] = str(player_mce)

    cmd = [
        "/app/target/release/cascadia-cli", str(games),
        "--selfplay-pool",
        "--init-weights", init_path,
        "--out", out_path,
        "--epsilon", str(epsilon),
    ]
    if temperature > 0:
        cmd += ["--temperature", str(temperature)]

    result = sp.run(cmd, capture_output=True, text=True, cwd="/app", env=env)
    if result.returncode != 0:
        raise RuntimeError(f"selfplay-pool failed: {result.stderr[-500:]}")

    # Parse reported stats for diagnostics
    samples = 0; elapsed = 0.0
    for line in result.stdout.splitlines():
        if line.startswith("SAMPLES="):
            samples = int(line.split("=", 1)[1])
        elif line.startswith("ELAPSED_SEC="):
            elapsed = float(line.split("=", 1)[1])
    print(f"[worker seed={seed}] {samples} samples in {elapsed:.1f}s")

    with open(out_path, "rb") as f:
        data = f.read()
    return data


@app.local_entrypoint()
def run(
    total_games: int = 50000,
    num_workers: int = 50,
    weights: str = "nnue_weights_v4opp_fsp_iter3.bin",
    opp_pool: str = "random,scarcity,preference,nnue_weights_mce93.bin,nnue_weights_mid_fsp_iter10.bin",
    out: str = "/tmp/selfplay_fsp_combined.bin",
    epsilon: float = 0.1,
    temperature: float = 0.0,
    seed_base: int = 4217,
    player_mce: int = 0,
):
    # Upload weights file + any .bin opponents to Modal volume (content-addressed).
    pool_entries = [e.strip() for e in opp_pool.split(",") if e.strip()]
    tag_entries = {"greedy", "random", "scarcity", "preference"}
    local_paths_to_upload = set()
    if os.path.exists(weights):
        local_paths_to_upload.add(weights)
    else:
        sys.exit(f"ERROR: weights file not found: {weights}")
    for e in pool_entries:
        if e.lower() in tag_entries:
            continue
        if not os.path.exists(e):
            sys.exit(f"ERROR: opp-pool entry not found: {e}")
        local_paths_to_upload.add(e)

    remote_name_for = {p: _content_addressed_name(p) for p in local_paths_to_upload}

    print(f"Syncing {len(local_paths_to_upload)} weight files to Modal volume...")
    t_up = time.time()
    try:
        existing = {e.path.lstrip("/") for e in weights_volume.iterdir("/")}
    except Exception:
        existing = set()
    to_upload = {p: n for p, n in remote_name_for.items() if n not in existing}
    if to_upload:
        with weights_volume.batch_upload() as batch:
            for local_path, remote_name in to_upload.items():
                batch.put_file(local_path, f"/{remote_name}")
    print(f"  Synced ({len(to_upload)} new, {len(remote_name_for) - len(to_upload)} cached) in {time.time()-t_up:.1f}s")

    # Resolve opp-pool names to remote basenames (for tags, pass through).
    pool_remote = []
    for e in pool_entries:
        if e.lower() in tag_entries:
            pool_remote.append(e)
        else:
            pool_remote.append(remote_name_for[e])

    init_remote = remote_name_for[weights]

    # Split games across workers; last worker absorbs remainder.
    games_per_worker = [total_games // num_workers] * num_workers
    for i in range(total_games - sum(games_per_worker)):
        games_per_worker[i] += 1

    print(f"\nDispatching {num_workers} workers — {total_games} total games")
    print(f"  weights: {weights} ({init_remote})")
    print(f"  opp pool: {pool_remote}")
    print(f"  epsilon: {epsilon}  temperature: {temperature}")
    if player_mce > 0:
        print(f"  player_mce: MCE({player_mce}) for player 0 (on-policy training)")

    t0 = time.time()
    futures = []
    for i, games in enumerate(games_per_worker):
        f = selfplay_worker.spawn(
            games=games,
            init_weight_name=init_remote,
            opp_pool_names=pool_remote,
            seed=seed_base + i * 7919,
            epsilon=epsilon,
            temperature=temperature,
            player_mce=player_mce,
        )
        futures.append((i, f))

    print(f"  dispatched in {time.time()-t0:.1f}s, awaiting results...")

    # Gather and concatenate shards. First shard keeps its MCV3 magic; subsequent
    # shards have their 4-byte magic stripped before appending.
    total_bytes = 0
    with open(out, "wb") as out_f:
        for idx, (i, fut) in enumerate(futures):
            data = fut.get()
            if idx == 0:
                if data[:4] != b"MCV3":
                    sys.exit(f"ERROR: shard {i} missing MCV3 magic")
                out_f.write(data)
            else:
                if data[:4] != b"MCV3":
                    print(f"  WARN: shard {i} missing MCV3 magic — writing as-is")
                    out_f.write(data)
                else:
                    out_f.write(data[4:])
            total_bytes += len(data)
            elapsed = time.time() - t0
            done = idx + 1
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} shards ({elapsed:.0f}s, {total_bytes/1024/1024:.1f} MB)")

    size = os.path.getsize(out)
    print(f"\nCombined: {size/1024/1024:.1f} MB → {out}")
    print(f"Total wall clock: {time.time()-t0:.0f}s")
    cpu_sec_estimate = num_workers * 8 * (time.time() - t0) / num_workers  # rough
    cost = cpu_sec_estimate * 0.000014 * num_workers
    print(f"Estimated cost: ~${cost:.2f}")
