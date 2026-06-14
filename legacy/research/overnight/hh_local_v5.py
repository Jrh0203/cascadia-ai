#!/usr/bin/env python3
"""Local head-to-head runner — interruptible, append-only JSONL, resumable.

Spawns one cascadia-cli per game sequentially, saving each game's per-player
breakdown to a JSONL file as soon as the game finishes. Safe to Ctrl-C at any
point; partial results are usable. Re-running with the same --jsonl-out skips
games already recorded (resume).

Each "game slot" is identified by (game_idx, rotation), giving 4 rotations per
sample for fair seat-counterbalancing. With --num-games 50, the runner plays 12
samples × 4 rotations = 48 games, then a final sample with 2 rotations to hit 50.

Usage:
    python3 overnight/hh_local_v5.py \\
        --strategy-a mce_wide_v1 --weights-a nnue_weights_v5sh_iter5.bin \\
        --strategy-b mce_wide_v1_b --weights-b nnue_weights_v4opp_modal_iter3.bin \\
        --binary target-mid-v5/release/cascadia-cli \\
        --num-games 50 \\
        --jsonl-out overnight/v5sh/hh_local.jsonl

To analyze partial results (any time):
    python3 overnight/hh_local_v5.py --summarize overnight/v5sh/hh_local.jsonl
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Stop flag set on SIGINT so the runner finishes the current game cleanly
# (subprocess receives the signal and dies; we then break the loop).
_STOP = {"flag": False}


def _on_sigint(signum, frame):
    _STOP["flag"] = True
    print("\n[hh_local] Ctrl-C received — finishing after current game.", flush=True)


PLAYER_RE = re.compile(
    r"SYMPLAYER p=(\d+) base=(\d+) bonus=(\d+) hab=(\d+) wl=(\d+) tok=(\d+) "
    r"bear=(\d+) elk=(\d+) salmon=(\d+) hawk=(\d+) fox=(\d+)"
)


def play_one_game(binary, seating_tags, seat_weights, seed_offset, env_extra):
    """Spawn cascadia-cli for one game. Returns dict with seating + players."""
    env = os.environ.copy()
    env["CASCADIA_SEAT_STRATEGIES"] = ":".join(seating_tags)
    env["CASCADIA_SEAT_WEIGHTS"] = ":".join(seat_weights)
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)
    env.update(env_extra)
    # default --weights points at any one of the seat weights (it's the
    # strategy-default fallback; per-seat weights take precedence at runtime).
    cmd = [binary, "1", "--nnue", "--weights", seat_weights[0]]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False, timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "elapsed": time.time() - t0,
                "seating": seating_tags}
    elapsed = time.time() - t0
    players = []
    # SYMPLAYER lines go to stdout in --nnue mode (not stderr).
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        m = PLAYER_RE.search(line)
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
        "seating": seating_tags,
        "players": players,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "stderr_tail": (result.stderr + " | " + result.stdout)[-500:] if not players else "",
    }


def load_completed(jsonl_path):
    """Return set of (sample_i, rot_i) tuples already recorded."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["sample_i"], rec["rot_i"]))
            except Exception:
                continue
    return done


def summarize(jsonl_path):
    """Read the JSONL and print aggregate stats per strategy tag."""
    if not os.path.exists(jsonl_path):
        print(f"no JSONL at {jsonl_path}")
        return
    stats = {}
    games = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            players = rec.get("players", [])
            if len(players) != 4:
                continue
            seating = rec["seating"]
            # Rank by base desc, tie-break on tokens (matches Modal HH).
            ranked = sorted(range(4),
                            key=lambda i: (-players[i]["base"], -players[i]["tok"]))
            ranks = [0] * 4
            for rk, idx in enumerate(ranked):
                ranks[idx] = rk + 1
            for seat in range(4):
                tag = seating[seat]
                pl = players[seat]
                s = stats.setdefault(tag, {"games": 0, "scores": [], "bonuses": [],
                                          "ranks": [], "wins": 0,
                                          "bear": [], "elk": [], "salmon": [],
                                          "hawk": [], "fox": []})
                s["games"] += 1
                s["scores"].append(pl["base"])
                s["bonuses"].append(pl["bonus"])
                s["ranks"].append(ranks[seat])
                if ranks[seat] == 1:
                    s["wins"] += 1
                for k in ("bear", "elk", "salmon", "hawk", "fox"):
                    s[k].append(pl[k])
            games += 1

    if games == 0:
        print("no completed games")
        return

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def stderr(xs):
        if len(xs) <= 1:
            return 0.0
        m = mean(xs)
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return (var / len(xs)) ** 0.5

    print("=" * 82)
    print(f"LOCAL HH RESULTS — {games} games, {sum(s['games'] for s in stats.values())} seat-games")
    print("=" * 82)
    print(f"\n{'Strategy':<18} {'Games':>5} {'WinRate':>8} {'MeanRank':>9} "
          f"{'MeanScore':>10} {'Bonus':>7} {'SE':>6}")
    print("-" * 82)
    for tag, s in stats.items():
        n = s["games"]
        win_rate = 100 * s["wins"] / n if n else 0
        print(f"{tag:<18} {n:>5} {win_rate:>7.1f}% {mean(s['ranks']):>9.2f} "
              f"{mean(s['scores']):>10.2f} {mean(s['bonuses']):>7.2f} "
              f"{stderr(s['scores']):>6.2f}")

    print(f"\n{'Strategy':<18} {'Bear':>7} {'Elk':>7} {'Salmon':>7} {'Hawk':>7} {'Fox':>7}")
    print("-" * 60)
    for tag, s in stats.items():
        if not s["scores"]:
            continue
        print(f"{tag:<18} "
              f"{mean(s['bear']):>7.2f} {mean(s['elk']):>7.2f} "
              f"{mean(s['salmon']):>7.2f} {mean(s['hawk']):>7.2f} "
              f"{mean(s['fox']):>7.2f}")

    # Pairwise delta if exactly 2 strategies
    if len(stats) == 2:
        tags = list(stats.keys())
        a, b = stats[tags[0]], stats[tags[1]]
        delta = mean(a["scores"]) - mean(b["scores"])
        # Approximate SE of difference: each side's SE assuming independence
        se_a = stderr(a["scores"])
        se_b = stderr(b["scores"])
        se_diff = (se_a ** 2 + se_b ** 2) ** 0.5
        sigma = delta / se_diff if se_diff > 0 else 0
        print(f"\nDelta ({tags[0]} - {tags[1]}): {delta:+.2f} pts "
              f"(SE_diff={se_diff:.3f}, ~{sigma:+.2f}σ)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy-a", default="mce_wide_v1")
    p.add_argument("--strategy-b", default="mce_wide_v1_b")
    p.add_argument("--weights-a", required=False)
    p.add_argument("--weights-b", required=False)
    p.add_argument("--binary", default="target-mid-v5/release/cascadia-cli")
    p.add_argument("--num-games", type=int, default=50,
                   help="Total games to play. Distributed as ceil(N/4) samples × 4 rotations; "
                        "the final sample may have fewer than 4 rotations to hit N exactly.")
    p.add_argument("--jsonl-out", required=False)
    p.add_argument("--env", default="MCE_LMR=1,MCE_DIVERSE_PREFILTER=1,MCE_MUTATE_EXPAND=24")
    p.add_argument("--parallel", type=int, default=3,
                   help="Concurrent games to run. Each game internally uses ~3 cores (sequential "
                        "halving's late rounds collapse parallelism). With --parallel 3 on a 10-core "
                        "machine, we saturate ~9 cores for ~2.5x throughput vs serial. Set to 1 to "
                        "disable parallelism. Set higher than 4 only if you have >12 cores.")
    p.add_argument("--summarize", default=None,
                   help="Skip play; just summarize an existing JSONL.")
    args = p.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return

    if not (args.weights_a and args.weights_b and args.jsonl_out):
        p.error("--weights-a, --weights-b, and --jsonl-out required for play mode")
    if not os.path.exists(args.binary):
        p.error(f"binary not found: {args.binary}")
    if not os.path.exists(args.weights_a):
        p.error(f"weights-a not found: {args.weights_a}")
    if not os.path.exists(args.weights_b):
        p.error(f"weights-b not found: {args.weights_b}")

    env_extra = {}
    for pair in args.env.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            env_extra[k.strip()] = v.strip()

    signal.signal(signal.SIGINT, _on_sigint)
    os.makedirs(os.path.dirname(os.path.abspath(args.jsonl_out)), exist_ok=True)

    completed = load_completed(args.jsonl_out)
    n_resume = len(completed)
    if n_resume:
        print(f"[hh_local] Resuming — {n_resume} games already in {args.jsonl_out}")

    # Schedule: ceil(N/4) samples × 4 rotations, last sample may be partial.
    rotations_per_sample = 4
    num_samples = (args.num_games + rotations_per_sample - 1) // rotations_per_sample
    print(f"[hh_local] Playing {args.num_games} games "
          f"({num_samples} samples × up to {rotations_per_sample} rotations, "
          f"parallel={args.parallel})")
    print(f"[hh_local] A: {args.strategy_a} = {args.weights_a}")
    print(f"[hh_local] B: {args.strategy_b} = {args.weights_b}")
    print(f"[hh_local] env: {env_extra}")
    print(f"[hh_local] writing to {args.jsonl_out}")

    base_seating = [args.strategy_a, args.strategy_b, args.strategy_a, args.strategy_b]
    base_weights = [args.weights_a, args.weights_b, args.weights_a, args.weights_b]

    # Build the full task list (sample_i, rot_i, seed_offset, seating, seat_weights)
    # honoring resume + games-target cap.
    tasks = []
    games_dispatched = 0
    games_total_target = args.num_games
    for sample_i in range(num_samples):
        for rot_i in range(rotations_per_sample):
            if games_dispatched >= games_total_target:
                break
            games_dispatched += 1
            if (sample_i, rot_i) in completed:
                continue
            seating = [base_seating[(rot_i + s) % 4] for s in range(4)]
            seat_weights = [base_weights[(rot_i + s) % 4] for s in range(4)]
            seed_offset = sample_i * 1000 + rot_i
            tasks.append((sample_i, rot_i, seed_offset, seating, seat_weights))

    print(f"[hh_local] {len(tasks)} games to play "
          f"({games_total_target - len(tasks)} skipped from resume)")

    out_f = open(args.jsonl_out, "a", buffering=1)  # line-buffered
    out_lock = threading.Lock()                     # serialize JSONL writes
    games_played = 0
    games_played_lock = threading.Lock()
    t_start = time.time()

    def task_runner(task):
        """Worker callable for one game. Returns (sample_i, rot_i, result)."""
        sample_i, rot_i, seed_offset, seating, seat_weights = task
        if _STOP["flag"]:
            return sample_i, rot_i, None
        t0 = time.time()
        result = play_one_game(args.binary, seating, seat_weights,
                               seed_offset, env_extra)
        result["sample_i"] = sample_i
        result["rot_i"] = rot_i
        result["seed_offset"] = seed_offset
        result["wall_elapsed"] = time.time() - t0
        return sample_i, rot_i, result

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(task_runner, task): task for task in tasks}
        for fut in as_completed(futures):
            if _STOP["flag"]:
                # Cancel anything that hasn't started yet; running games will
                # complete and their results will be written below.
                for f2 in futures:
                    if not f2.done() and not f2.running():
                        f2.cancel()
            try:
                sample_i, rot_i, result = fut.result()
            except Exception as e:
                print(f"[hh_local] task failed: {e}", flush=True)
                continue
            if result is None:
                continue
            with out_lock:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
            with games_played_lock:
                games_played += 1
                gp_now = games_played
            scores = ([pl["base"] for pl in result.get("players", [])]
                      if result.get("players") else "ERROR")
            done_so_far = n_resume + gp_now
            total_elapsed = time.time() - t_start
            avg_throughput = gp_now / total_elapsed if total_elapsed > 0 else 0
            remaining = games_total_target - done_so_far
            eta_min = (remaining / avg_throughput / 60) if avg_throughput > 0 else 0
            print(f"  → s{sample_i+1} r{rot_i}: {scores} "
                  f"({result.get('wall_elapsed', 0):.0f}s game-wall)  "
                  f"[{done_so_far}/{games_total_target}, "
                  f"throughput={avg_throughput*60:.1f} games/min, ETA {eta_min:.0f}m]",
                  flush=True)

    out_f.close()
    print(f"\n[hh_local] Played {games_played} new games "
          f"(total {n_resume + games_played} in JSONL).")
    summarize(args.jsonl_out)


if __name__ == "__main__":
    main()
