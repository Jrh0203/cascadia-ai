#!/usr/bin/env python3
"""4-player round-robin tournament — Modal parallel.

Plays all-4-strategies-per-game tournament games on Modal in parallel.
Each game uses 4 distinct mce_wide_v1-class tags (a/b/c/d) so each seat gets
its own NNUE weights but identical algorithm.

Saves per-game JSONL identical to tournament_v5.py format so the existing
analysis script (overnight/tournament_analysis.py) can consume it.

Usage:
    python3 -m modal run overnight/tournament_modal.py \\
      --strategies-csv 'v5sh_iter40=nnue_weights_v5sh_iter40.bin,v4opp=nnue_weights_v4opp_modal_iter3.bin,mid_fsp_iter10=nnue_weights_mid_fsp_iter10.bin,mce93=nnue_weights_mce93.bin' \\
      --rounds 30 \\
      --jsonl-out overnight/v5sh/tournament.jsonl
"""
import hashlib
import json
import modal
import os
import re
import sys
import time

app = modal.App("cascadia-tournament-v5")

weights_volume = modal.Volume.from_name("cascadia-weights-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "gcc", "pkg-config", "libssl-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .add_local_dir("crates", remote_path="/app/crates", copy=True)
    .add_local_file("Cargo.toml", remote_path="/app/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", remote_path="/app/Cargo.lock", copy=True)
    .run_commands(
        "cd /app && cargo build --release --features mid-features,v4-opp,v5-feat --bin cascadia-cli",
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=1800,
    volumes={"/weights": weights_volume},
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=5.0),
)
def play_one_game(
    seat_tags: list,            # 4 strategy tags
    seat_weight_names: list,    # 4 basenames in /weights/
    seat_strategy_names: list,  # 4 human-readable names (for analysis)
    round_i: int,
    rot_i: int,
    seed_offset: int,
    extra_env: dict = None,
) -> dict:
    """Run one tournament game on Modal. Returns dict with full per-seat detail."""
    import subprocess as sp

    seat_paths = [f"/weights/{n}" for n in seat_weight_names]
    default_path = seat_paths[0]

    env = os.environ.copy()
    env["CASCADIA_SEAT_STRATEGIES"] = ":".join(seat_tags)
    env["CASCADIA_SEAT_WEIGHTS"] = ":".join(seat_paths)
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    cmd = ["/app/target/release/cascadia-cli", "1", "--nnue", "--weights", default_path]
    t0 = time.time()
    result = sp.run(cmd, capture_output=True, text=True, cwd="/app", env=env)
    elapsed = time.time() - t0

    player_re = re.compile(
        r"SYMPLAYER p=(\d+) base=(\d+) bonus=(\d+) hab=(\d+) wl=(\d+) tok=(\d+) "
        r"bear=(\d+) elk=(\d+) salmon=(\d+) hawk=(\d+) fox=(\d+)"
    )
    players = []
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        m = player_re.search(line)
        if m:
            players.append({
                "p": int(m.group(1)),
                "base": int(m.group(2)),
                "bonus": int(m.group(3)),
                "hab": int(m.group(4)),
                "wl": int(m.group(5)),
                "tok": int(m.group(6)),
                "bear": int(m.group(7)),
                "elk": int(m.group(8)),
                "salmon": int(m.group(9)),
                "hawk": int(m.group(10)),
                "fox": int(m.group(11)),
            })

    return {
        "round_i": round_i,
        "rot_i": rot_i,
        "seed_offset": seed_offset,
        "seat_tags": seat_tags,
        "seat_weights": seat_weight_names,
        "seat_names": seat_strategy_names,
        "players": players,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "stderr_tail": (result.stderr + " | " + result.stdout)[-500:] if not players else "",
    }


def _content_addressed_name(local_path: str) -> str:
    h = hashlib.sha256()
    with open(local_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()[:8]
    base = os.path.basename(local_path)
    stem, ext = os.path.splitext(base)
    return f"{stem}-{digest}{ext}"


@app.local_entrypoint()
def run(
    strategies_csv: str,
    rounds: int = 30,
    jsonl_out: str = "overnight/v5sh/tournament.jsonl",
    env: str = "MCE_LMR=1,MCE_DIVERSE_PREFILTER=1,MCE_MUTATE_EXPAND=24",
):
    """Dispatch a 4-strategy round-robin tournament on Modal.

    `strategies_csv`: 4 entries like "name1=path1,name2=path2,name3=path3,name4=path4"
    `rounds`: total games = rounds × 4 cyclic rotations.
    """
    # Parse strategies (positional order matters — strategy at index i uses tag SEAT_TAGS[i])
    strats = []
    for spec in strategies_csv.split(","):
        if "=" not in spec:
            sys.exit(f"Bad strategy spec: {spec}")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            sys.exit(f"weights not found: {path}")
        strats.append((name.strip(), path.strip()))
    if len(strats) != 4:
        sys.exit(f"Need exactly 4 strategies, got {len(strats)}")

    SEAT_TAGS = ["mce_wide_v1", "mce_wide_v1_b", "mce_wide_v1_c", "mce_wide_v1_d"]

    env_extra = {}
    for pair in env.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            env_extra[k.strip()] = v.strip()

    # Upload weights to Modal volume (content-addressed; skip if already there)
    remote_for = {p: _content_addressed_name(p) for _, p in strats}
    print(f"Syncing {len(remote_for)} weight files to volume...")
    t_up = time.time()
    try:
        existing = {e.path.lstrip("/") for e in weights_volume.iterdir("/")}
    except Exception:
        existing = set()
    to_upload = {p: n for p, n in remote_for.items() if n not in existing}
    if to_upload:
        with weights_volume.batch_upload() as batch:
            for local_path, remote_name in to_upload.items():
                batch.put_file(local_path, f"/{remote_name}")
    print(f"  {len(to_upload)} new uploaded, {len(remote_for) - len(to_upload)} cached "
          f"({time.time() - t_up:.1f}s)")

    # Build the task list — each round has 4 cyclic-rotation games
    total_games = rounds * 4
    tasks = []
    for round_i in range(rounds):
        for rot_i in range(4):
            seat_strats = [strats[(s + rot_i) % 4] for s in range(4)]
            seat_names = [name for name, _ in seat_strats]
            seat_weight_names = [remote_for[path] for _, path in seat_strats]
            seed_offset = round_i * 1000 + rot_i
            tasks.append((round_i, rot_i, seed_offset, seat_names, seat_weight_names))

    # Resume support: drop tasks already in JSONL
    completed = set()
    if os.path.exists(jsonl_out):
        with open(jsonl_out) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add((rec["round_i"], rec["rot_i"]))
                except Exception:
                    pass
    if completed:
        before = len(tasks)
        tasks = [t for t in tasks if (t[0], t[1]) not in completed]
        print(f"Resuming: {before - len(tasks)} games already in {jsonl_out}, {len(tasks)} new")

    print(f"Tournament: 4 strategies × {rounds} rounds × 4 rotations = {total_games} games "
          f"({len(tasks)} to dispatch)")
    print(f"Strategies:")
    for i, (name, path) in enumerate(strats):
        print(f"  [seat-tag {SEAT_TAGS[i]}] {name} → {path}")
    print()

    os.makedirs(os.path.dirname(os.path.abspath(jsonl_out)), exist_ok=True)
    out_f = open(jsonl_out, "a", buffering=1)

    t0 = time.time()
    futures = []
    for round_i, rot_i, seed_offset, seat_names, seat_weight_names in tasks:
        f = play_one_game.spawn(
            seat_tags=SEAT_TAGS,
            seat_weight_names=seat_weight_names,
            seat_strategy_names=seat_names,
            round_i=round_i,
            rot_i=rot_i,
            seed_offset=seed_offset,
            extra_env=env_extra or None,
        )
        futures.append(f)

    print(f"Dispatched {len(futures)} games in {time.time() - t0:.1f}s, awaiting results...")

    done = 0
    for fut in futures:
        try:
            result = fut.get()
        except Exception as e:
            print(f"  task failed: {e}")
            continue
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()
        done += 1
        scores = ([pl["base"] for pl in result.get("players", [])]
                  if result.get("players") else "ERROR")
        elapsed = time.time() - t0
        print(f"  r{result['round_i']+1} rot{result['rot_i']}: "
              f"seats={result['seat_names']}  scores={scores}  "
              f"({result.get('elapsed', 0):.0f}s game)  "
              f"[{done}/{len(futures)} | {elapsed:.0f}s wall]", flush=True)

    out_f.close()
    print(f"\nTournament complete. {done} games written to {jsonl_out}")
    print(f"Total wall: {time.time() - t0:.0f}s. Cost: ~${(time.time() - t0) * len(futures) * 8 * 0.000014:.2f}")
    print(f"\nRun analysis: python3 overnight/tournament_analysis.py {jsonl_out}")
