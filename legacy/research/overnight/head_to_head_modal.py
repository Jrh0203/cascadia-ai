"""Modal-dispatched head-to-head tournament.

Port of overnight/head_to_head.py that runs each game as a separate Modal
worker, giving ~10× wall-clock speedup over local execution. Each worker is
8 vCPUs, one HH game per worker, all 20 games fire in parallel.

Weights files are uploaded ONCE per run to a persistent Modal volume
(`cascadia-weights-cache`) and referenced by filename in every spawn call,
so per-game payload stays tiny and subsequent runs with the same weights
skip the upload entirely.

Usage (from local):
    modal run overnight/head_to_head_modal.py \
        --strategies "mce_new,mce_anchor,mce_anchor,mce_anchor" \
        --strategy-weights "mce_new=nnue_weights_sym_pool_iter1.bin,mce_anchor=nnue_weights_mce93.bin" \
        --game-samples 5

Cost: ~$0.011 per game × 20 games = ~$0.22 per N=20 tournament.
"""

import hashlib
import modal
import os
import re
import sys
import time
from collections import defaultdict

app = modal.App("cascadia-hh")

# Persistent weights cache. Each unique local path is uploaded under a
# content-addressed name so re-runs with the same weights skip upload entirely,
# and an edited weights file (same local path, different bytes) gets a new
# remote name rather than overwriting the old one.
weights_volume = modal.Volume.from_name("cascadia-weights-cache", create_if_missing=True)

# Match the image used in modal_collect.py so we hit its cache when available.
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
        # Build with mid-features + v4-opp so v4opp weights can be loaded.
        # Legacy (mce93) and mid_fsp_iter10 weights are forward-compatible via
        # zero-padding for missing columns.
        "cd /app && cargo build --release --features mid-features,v4-opp --bin cascadia-cli",
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=4096,
    timeout=1800,
    volumes={"/weights": weights_volume},
)
def hh_single_game(
    seating: list,            # list[str] of 4 strategy tags in seat order
    seat_weight_names: list,  # list[str], one per seat — basename in /weights/
    default_weight_name: str, # basename in /weights/ used as the --weights arg
    seed_offset: int,
    extra_env: dict = None,   # additional env vars (e.g. MCE_DIVERSE_PREFILTER=1)
) -> dict:
    """Run one HH game on Modal. Returns {'seating': [...], 'players': [...]}.

    All weights are expected at /weights/<name>, uploaded once by the caller.
    Per-spawn payload is just filenames, keeping dispatch fast.
    """
    import subprocess as sp

    default_path = f"/weights/{default_weight_name}"
    seat_paths = [f"/weights/{n}" for n in seat_weight_names]

    env = os.environ.copy()
    env["CASCADIA_SEAT_STRATEGIES"] = ":".join(seating)
    env["CASCADIA_SEAT_WEIGHTS"] = ":".join(seat_paths)
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    cmd = [
        "/app/target/release/cascadia-cli", "1",
        "--nnue", "--weights", default_path,
    ]
    result = sp.run(cmd, capture_output=True, text=True, cwd="/app", env=env)

    player_re = re.compile(
        r"SYMPLAYER p=(\d+) base=(\d+) bonus=(\d+) hab=(\d+) wl=(\d+) tok=(\d+) "
        r"bear=(\d+) elk=(\d+) salmon=(\d+) hawk=(\d+) fox=(\d+)"
    )
    players = []
    for line in result.stderr.splitlines():
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
    return {"seating": seating, "players": players, "stderr_tail": result.stderr[-500:]}


def _content_addressed_name(local_path: str) -> str:
    """basename + first 8 chars of sha256 so identical content → same remote
    name (skipped upload), edited content → new remote name (fresh upload)."""
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
    strategies: str,
    strategy_weights: str,
    game_samples: int = 5,
    weights: str = "nnue_weights_mce93.bin",
    env: str = "",
    ensemble_weights: str = "",
):
    """Dispatch HH tournament. `strategies` and `strategy_weights` match the
    format of overnight/head_to_head.py.

    `ensemble_weights` (optional): comma-separated local .bin paths. Uploaded
    to volume alongside strategy weights; their remote paths are joined into
    CASCADIA_ENS_PATHS in the worker env. Useful for `mce_wide_ens_v1` tag
    which reads CASCADIA_ENS_PATHS to activate `MCE_PREFILTER_ENSEMBLE`.
    """
    import numpy as np  # local-side only

    strat_list = [s.strip() for s in strategies.split(",")]
    if len(strat_list) != 4:
        sys.exit(f"Need exactly 4 strategies, got {len(strat_list)}")

    # Parse --env "KEY=VAL,KEY2=VAL2" into dict for worker env vars
    game_extra_env = {}
    if env:
        for pair in env.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                game_extra_env[k.strip()] = v.strip()
        print(f"Extra env for workers: {game_extra_env}")

    # Parse strategy_weights: "tag1=path1,tag2=path2"
    strat_weight_paths = {}
    for pair in strategy_weights.split(","):
        k, v = pair.split("=", 1)
        strat_weight_paths[k.strip()] = v.strip()

    # Parse ensemble_weights (optional extra files to upload + wire as env).
    ensemble_local_paths = []
    if ensemble_weights:
        for p in ensemble_weights.split(","):
            p = p.strip()
            if not p:
                continue
            if not os.path.exists(p):
                sys.exit(f"ERROR: ensemble file not found: {p}")
            ensemble_local_paths.append(p)

    # Collect the unique local paths we need on the volume.
    unique_local_paths = set(strat_weight_paths.values())
    unique_local_paths.update(ensemble_local_paths)
    if os.path.exists(weights):
        unique_local_paths.add(weights)

    # Content-addressed remote names — same bytes → same name → volume dedupe.
    remote_name_for = {p: _content_addressed_name(p) for p in unique_local_paths}

    # Wire ensemble paths into worker env (only used by mce_wide_ens_v1 tag).
    if ensemble_local_paths:
        remote_ens_paths = ",".join(
            f"/weights/{remote_name_for[p]}" for p in ensemble_local_paths
        )
        game_extra_env["CASCADIA_ENS_PATHS"] = remote_ens_paths
        print(f"Ensemble paths for workers: CASCADIA_ENS_PATHS={remote_ens_paths}")

    # Upload any weights missing from the volume. We check what's already
    # present and only upload the missing ones — unchanged weights on a repeat
    # run cost zero bandwidth (same bytes → same content-addressed name →
    # already on volume → skip).
    print(f"Syncing {len(unique_local_paths)} weight files to Modal volume...")
    t_up = time.time()
    try:
        existing = {e.path.lstrip("/") for e in weights_volume.iterdir("/")}
    except Exception:
        existing = set()
    to_upload = {p: n for p, n in remote_name_for.items() if n not in existing}
    skipped = len(remote_name_for) - len(to_upload)
    if to_upload:
        with weights_volume.batch_upload() as batch:
            for local_path, remote_name in to_upload.items():
                batch.put_file(local_path, f"/{remote_name}")
    print(f"Sync done in {time.time() - t_up:.1f}s "
          f"(uploaded {len(to_upload)}, skipped {skipped}: "
          f"{[os.path.basename(p) for p in remote_name_for]})")

    default_weight_name = remote_name_for.get(weights) or next(iter(remote_name_for.values()))

    # Build rotations (cyclic shifts, matches head_to_head.py)
    rotations = []
    for rot in range(4):
        rotations.append([strat_list[(rot + seat) % 4] for seat in range(4)])

    total_games = game_samples * len(rotations)
    print(f"Tournament: 4 strategies, {game_samples} samples × 4 rotations "
          f"= {total_games} games (Modal parallel dispatch)")
    print(f"Strategies: {strat_list}")
    for i, r in enumerate(rotations):
        print(f"  rot{i}: {r}")
    print()

    # Dispatch all games in parallel — just filenames now, no bytes.
    t0 = time.time()
    futures = []
    for sample_i in range(game_samples):
        for rot_i, seating in enumerate(rotations):
            seat_names = [
                remote_name_for[strat_weight_paths.get(s, weights)]
                for s in seating
            ]
            f = hh_single_game.spawn(
                seating=seating,
                seat_weight_names=seat_names,
                default_weight_name=default_weight_name,
                seed_offset=sample_i * 1000,
                extra_env=game_extra_env or None,
            )
            futures.append((sample_i, rot_i, seating, f))

    print(f"Dispatched {len(futures)} games in {time.time() - t0:.1f}s, awaiting results...")

    # Gather
    results = []
    for sample_i, rot_i, seating, f in futures:
        r = f.get()
        results.append((sample_i, rot_i, seating, r))
        players = r["players"]
        if len(players) == 4:
            scores = [players[i]["base"] for i in range(4)]
        else:
            scores = f"<{len(players)} players>"
        elapsed = time.time() - t0
        print(f"  rot{rot_i} sample{sample_i+1}: {scores} ({elapsed:.0f}s elapsed)")

    # Aggregate stats per strategy tag
    stats = {s: {"games": 0, "scores": [], "bonuses": [], "ranks": [],
                 "bear": [], "elk": [], "salmon": [], "hawk": [], "fox": [],
                 "wins": 0} for s in strat_list}

    for _, _, seating, r in results:
        players = r["players"]
        if len(players) != 4:
            continue
        # Rank by base desc, tie-break on tokens
        ranked = sorted(range(4), key=lambda i: (-players[i]["base"], -players[i]["tok"]))
        ranks = [0] * 4
        for rk, idx in enumerate(ranked):
            ranks[idx] = rk + 1
        for seat in range(4):
            strat = seating[seat]
            pl = players[seat]
            stats[strat]["games"] += 1
            stats[strat]["scores"].append(pl["base"])
            stats[strat]["bonuses"].append(pl["bonus"])
            stats[strat]["ranks"].append(ranks[seat])
            if ranks[seat] == 1:
                stats[strat]["wins"] += 1
            for k in ["bear", "elk", "salmon", "hawk", "fox"]:
                stats[strat][k].append(pl[k])

    # Report
    print("\n" + "=" * 82)
    print("TOURNAMENT RESULTS")
    print("=" * 82)
    print(f"\n{'Strategy':<14} {'Games':>5} {'WinRate':>8} {'MeanRank':>9} "
          f"{'MeanScore':>9} {'Bonus':>7} {'SE':>6}")
    print("-" * 82)
    for s in strat_list:
        st = stats[s]
        if st["games"] == 0:
            print(f"{s:<14} {'?':>5}")
            continue
        n = st["games"]
        win_rate = 100 * st["wins"] / n
        mean_rank = np.mean(st["ranks"])
        mean_score = np.mean(st["scores"])
        mean_bonus = np.mean(st["bonuses"])
        se = np.std(st["scores"], ddof=1) / np.sqrt(n) if n > 1 else 0.0
        print(f"{s:<14} {n:>5} {win_rate:>7.1f}% {mean_rank:>9.2f} "
              f"{mean_score:>9.2f} {mean_bonus:>7.2f} {se:>6.2f}")

    print(f"\n{'Strategy':<14} {'Bear':>7} {'Elk':>7} {'Salmon':>7} {'Hawk':>7} {'Fox':>7}")
    print("-" * 60)
    for s in strat_list:
        st = stats[s]
        if not st["scores"]:
            continue
        print(f"{s:<14} "
              f"{np.mean(st['bear']):>7.2f} {np.mean(st['elk']):>7.2f} "
              f"{np.mean(st['salmon']):>7.2f} {np.mean(st['hawk']):>7.2f} "
              f"{np.mean(st['fox']):>7.2f}")

    print(f"\n{'Strategy':<14} {'Rank1':>7} {'Rank2':>7} {'Rank3':>7} {'Rank4':>7}")
    print("-" * 50)
    for s in strat_list:
        st = stats[s]
        if not st["ranks"]:
            continue
        rank_hist = np.bincount(st["ranks"], minlength=5)[1:5]
        total = sum(rank_hist)
        pcts = [100 * c / max(total, 1) for c in rank_hist]
        print(f"{s:<14} "
              f"{pcts[0]:>6.1f}% {pcts[1]:>6.1f}% {pcts[2]:>6.1f}% {pcts[3]:>6.1f}%")

    elapsed = time.time() - t0
    estimated_cost = len(futures) * 100 * 8 * 0.000014  # ~100s per game × 8 vCPUs × $/s
    print(f"\nWall clock: {elapsed:.0f}s  Estimated cost: ~${estimated_cost:.2f}")
