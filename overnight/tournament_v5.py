#!/usr/bin/env python3
"""4-player round-robin tournament runner.

Plays games where all 4 seats have DIFFERENT strategies (each with own NNUE
weights). All seats use the same algorithm (mce_wide_v1's R=600 + LMR + diverse
prefilter), only weights differ — so any score difference is attributable to
the weights, not algorithm/budget confounds.

Each "round" = 4 cyclic-rotation games so each strategy plays each seat
position equally. Run N rounds for 4N games total.

Output JSONL: one record per game with seating, players' scores, wildlife
breakdown, etc. Resumable (re-running with same --jsonl-out skips done games).

Usage:
    python3 overnight/tournament_v5.py \\
      --strategy 'v5sh_iter40=nnue_weights_v5sh_iter40.bin' \\
      --strategy 'v4opp=nnue_weights_v4opp_modal_iter3.bin' \\
      --strategy 'mid_fsp_iter10=nnue_weights_mid_fsp_iter10.bin' \\
      --strategy 'mce93=nnue_weights_mce93.bin' \\
      --rounds 15 \\
      --jsonl-out overnight/v5sh/tournament.jsonl
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

_STOP = {"flag": False}


def _on_sigint(signum, frame):
    _STOP["flag"] = True
    print("\n[tournament] Ctrl-C — finishing after current game.", flush=True)


PLAYER_RE = re.compile(
    r"SYMPLAYER p=(\d+) base=(\d+) bonus=(\d+) hab=(\d+) wl=(\d+) tok=(\d+) "
    r"bear=(\d+) elk=(\d+) salmon=(\d+) hawk=(\d+) fox=(\d+)"
)


def play_one_game(binary, seat_tags, seat_weights, seed_offset, env_extra):
    env = os.environ.copy()
    env["CASCADIA_SEAT_STRATEGIES"] = ":".join(seat_tags)
    env["CASCADIA_SEAT_WEIGHTS"] = ":".join(seat_weights)
    env["CASCADIA_SEED_OFFSET"] = str(seed_offset)
    env.update(env_extra)
    cmd = [binary, "1", "--nnue", "--weights", seat_weights[0]]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False, timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "elapsed": time.time() - t0}
    elapsed = time.time() - t0
    players = []
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
        "seat_tags": seat_tags,
        "seat_weights": [os.path.basename(w) for w in seat_weights],
        "players": players,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "stderr_tail": (result.stderr + " | " + result.stdout)[-500:] if not players else "",
    }


def load_completed(jsonl_path):
    """Return set of (round_i, rot_i) tuples already in the JSONL."""
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
                done.add((rec["round_i"], rec["rot_i"]))
            except Exception:
                continue
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", action="append", required=True,
                   help='Strategy spec: "name=weights_path". Pass --strategy multiple times '
                        '(must be exactly 4 strategies for a 4-player round-robin).')
    p.add_argument("--binary", default="target-mid-v5/release/cascadia-cli")
    p.add_argument("--rounds", type=int, default=15,
                   help="Number of rounds. Each round = 4 cyclic-rotation games (so all "
                        "strategies play each seat position equally per round). Total games = 4 × rounds.")
    p.add_argument("--jsonl-out", required=True)
    p.add_argument("--env", default="MCE_LMR=1,MCE_DIVERSE_PREFILTER=1,MCE_MUTATE_EXPAND=24")
    args = p.parse_args()

    # Parse strategies
    strategies = []
    for spec in args.strategy:
        if "=" not in spec:
            sys.exit(f"strategy spec missing '=': {spec}")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            sys.exit(f"weights not found for {name}: {path}")
        strategies.append((name.strip(), path.strip()))
    if len(strategies) != 4:
        sys.exit(f"need exactly 4 strategies, got {len(strategies)}")

    # Each strategy gets a distinct algorithm-tag alias so per-seat dispatch works.
    # All 4 tags are clones of mce_wide_v1 — same algorithm, just different names so
    # the CLI can route each seat's weights via CASCADIA_SEAT_WEIGHTS.
    SEAT_TAGS = ["mce_wide_v1", "mce_wide_v1_b", "mce_wide_v1_c", "mce_wide_v1_d"]

    env_extra = {}
    for pair in args.env.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            env_extra[k.strip()] = v.strip()

    if not os.path.exists(args.binary):
        sys.exit(f"binary not found: {args.binary}")

    signal.signal(signal.SIGINT, _on_sigint)
    os.makedirs(os.path.dirname(os.path.abspath(args.jsonl_out)), exist_ok=True)

    completed = load_completed(args.jsonl_out)
    n_resume = len(completed)

    rotations_per_round = 4
    total_games = args.rounds * rotations_per_round
    print(f"[tournament] {len(strategies)} strategies × {args.rounds} rounds × "
          f"{rotations_per_round} rotations = {total_games} games")
    print(f"[tournament] Strategies:")
    for i, (name, path) in enumerate(strategies):
        print(f"  [{i}] {name} -- {path}")
    print(f"[tournament] Resume: {n_resume} games already done")
    print(f"[tournament] env: {env_extra}")
    print(f"[tournament] writing to {args.jsonl_out}")

    out_f = open(args.jsonl_out, "a", buffering=1)
    games_played = 0
    t_start = time.time()

    for round_i in range(args.rounds):
        if _STOP["flag"]:
            break
        for rot_i in range(rotations_per_round):
            if _STOP["flag"]:
                break
            if (round_i, rot_i) in completed:
                continue

            # Cyclic seat assignment: seat[s] gets strategies[(s + rot_i) % 4]
            seat_strategies = [strategies[(s + rot_i) % 4] for s in range(4)]
            seat_names = [name for name, _ in seat_strategies]
            seat_weights = [path for _, path in seat_strategies]

            seed_offset = round_i * 1000 + rot_i
            t0 = time.time()
            print(f"[tournament] round {round_i+1}/{args.rounds} rot{rot_i} "
                  f"(seed={seed_offset})  seats={seat_names}", flush=True)
            result = play_one_game(args.binary, SEAT_TAGS, seat_weights,
                                   seed_offset, env_extra)
            result["round_i"] = round_i
            result["rot_i"] = rot_i
            result["seed_offset"] = seed_offset
            result["seat_names"] = seat_names  # name → seat mapping for analysis
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            games_played += 1
            elapsed = time.time() - t0
            scores = ([pl["base"] for pl in result.get("players", [])]
                      if result.get("players") else "ERROR")
            done_so_far = n_resume + games_played
            total_elapsed = time.time() - t_start
            avg_per_game = total_elapsed / games_played if games_played else 0
            remaining = total_games - done_so_far
            eta_min = (remaining * avg_per_game) / 60 if avg_per_game else 0
            print(f"  → {scores} ({elapsed:.0f}s)  "
                  f"[{done_so_far}/{total_games}, ETA {eta_min:.0f}m]", flush=True)

    out_f.close()
    print(f"\n[tournament] Done. Played {games_played} new, total in JSONL "
          f"{n_resume + games_played}.")


if __name__ == "__main__":
    main()
