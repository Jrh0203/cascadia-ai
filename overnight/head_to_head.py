"""Head-to-head round-robin: 4 different strategies per game, rotating seats.

Rotates seats so each strategy plays each seat an equal number of times,
neutralizing seat position effects (player 0 drafts first each turn).

Measures per strategy:
- Win rate (% of games ranked 1st)
- Mean rank (1-4, lower is better)
- Mean score
- Per-animal breakdown

Usage:
    python3 overnight/head_to_head.py --strategies greedy,nnue,mce,mce_sr \\
        --game-samples 10  # 40 games total (4 rotations × 10)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
import numpy as np


CLI = "./target/release/cascadia-cli"


def run_one_game(strategies_in_seat_order, weights, extra_env=None, seat_weights=None):
    """Run one game with strategies assigned to seats. Returns list of 4 dicts.

    If seat_weights is provided (list of 4 paths), each seat uses its own
    NNUE weights. Otherwise all seats share the same `weights` arg.
    """
    tags = ":".join(strategies_in_seat_order)
    cmd = [CLI, "1", "--nnue", "--weights", weights]
    env = os.environ.copy()
    env["CASCADIA_SEAT_STRATEGIES"] = tags
    if seat_weights is not None:
        assert len(seat_weights) == 4
        env["CASCADIA_SEAT_WEIGHTS"] = ":".join(seat_weights)
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out, err = proc.communicate()

    player_re = re.compile(
        r"SYMPLAYER p=(\d+) base=(\d+) bonus=(\d+) hab=(\d+) wl=(\d+) tok=(\d+) "
        r"bear=(\d+) elk=(\d+) salmon=(\d+) hawk=(\d+) fox=(\d+)"
    )
    players = []
    for line in err.decode(errors="replace").splitlines():
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
    return players


def run_tournament(strategies, game_samples, weights, strategy_weights=None):
    """Run rotating-seat tournament. Each strategy plays each seat `game_samples` times.

    If `strategy_weights` is provided (dict mapping strategy_tag → weights_path),
    each strategy uses its own NNUE weights. Otherwise all share `weights`.
    """
    n = len(strategies)
    assert n == 4, "Currently supports exactly 4 strategies"

    # Rotation generates n different seat assignments (cyclic shifts)
    # Rotation i: strategies[(seat - i) % n] sits in each seat
    rotations = []
    for rot in range(n):
        seating = [strategies[(rot + seat) % n] for seat in range(n)]
        rotations.append(seating)

    total_games = game_samples * len(rotations)
    print(f"Tournament: {n} strategies, {game_samples} game-samples × {len(rotations)} rotations "
          f"= {total_games} games")
    print(f"Strategies: {strategies}")
    print(f"Rotations:")
    for i, r in enumerate(rotations):
        print(f"  rot{i}: {r}")
    print()

    # Per-strategy stat accumulators (across all seats they occupied)
    stats = {s: {"games": 0, "scores": [], "bonuses": [], "ranks": [],
                 "bear": [], "elk": [], "salmon": [], "hawk": [], "fox": [],
                 "wins": 0} for s in strategies}

    t0 = time.time()
    game_idx = 0
    for sample_i in range(game_samples):
        for rot_i, seating in enumerate(rotations):
            game_idx += 1
            # Use seed offset to get distinct games per sample but same across rotations
            # so rotations see the same scenario
            env = {"CASCADIA_SEED_OFFSET": str(sample_i * 1000)}
            seat_weights = None
            if strategy_weights is not None:
                seat_weights = [strategy_weights.get(s, weights) for s in seating]
            players = run_one_game(seating, weights, extra_env=env, seat_weights=seat_weights)
            if len(players) != 4:
                print(f"  WARN: game {game_idx} had {len(players)} players, skipping")
                continue

            # Rank players by base score (desc), tie-break on nature tokens
            # (official Cascadia tie-break rule).
            ranked = sorted(range(4), key=lambda i: (-players[i]["base"], -players[i]["tok"]))
            ranks = [0] * 4
            for r, idx in enumerate(ranked):
                ranks[idx] = r + 1  # 1 = best

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

            elapsed = time.time() - t0
            rate = game_idx / elapsed
            eta = (total_games - game_idx) / max(rate, 0.001)
            print(f"  Game {game_idx}/{total_games} (rot{rot_i}, sample{sample_i+1}): "
                  f"{[players[i]['base'] for i in range(4)]} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    return stats


def report(strategies, stats):
    print("\n" + "=" * 82)
    print("TOURNAMENT RESULTS")
    print("=" * 82)

    print(f"\n{'Strategy':<14} {'Games':>5} {'WinRate':>8} {'MeanRank':>9} {'MeanScore':>9} {'Bonus':>7} {'SE':>6}")
    print("-" * 82)
    for s in strategies:
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
        print(f"{s:<14} {n:>5} {win_rate:>7.1f}% {mean_rank:>9.2f} {mean_score:>9.2f} {mean_bonus:>7.2f} {se:>6.2f}")

    print(f"\n{'Strategy':<14} {'Bear':>7} {'Elk':>7} {'Salmon':>7} {'Hawk':>7} {'Fox':>7}")
    print("-" * 60)
    for s in strategies:
        st = stats[s]
        if not st["scores"]: continue
        print(f"{s:<14} "
              f"{np.mean(st['bear']):>7.2f} {np.mean(st['elk']):>7.2f} "
              f"{np.mean(st['salmon']):>7.2f} {np.mean(st['hawk']):>7.2f} "
              f"{np.mean(st['fox']):>7.2f}")

    # Rank distribution
    print(f"\n{'Strategy':<14} {'Rank1':>7} {'Rank2':>7} {'Rank3':>7} {'Rank4':>7}")
    print("-" * 50)
    for s in strategies:
        st = stats[s]
        if not st["ranks"]: continue
        rank_hist = np.bincount(st["ranks"], minlength=5)[1:5]
        total = sum(rank_hist)
        pcts = [100 * c / max(total, 1) for c in rank_hist]
        print(f"{s:<14} "
              f"{pcts[0]:>6.1f}% {pcts[1]:>6.1f}% {pcts[2]:>6.1f}% {pcts[3]:>6.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", required=True,
                   help="Comma-separated list of 4 strategy tags "
                        "(e.g., greedy,nnue,mce,mce_sr)")
    p.add_argument("--game-samples", type=int, default=10,
                   help="Games per rotation (total games = this × 4)")
    p.add_argument("--weights", default="nnue_weights_v9_iter14.bin")
    p.add_argument("--strategy-weights", default=None,
                   help="Per-strategy weights, format: tag1=path1,tag2=path2,... "
                        "Use for NNUE head-to-head where each strategy uses its own weights.")
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    if len(strategies) != 4:
        sys.exit(f"Need exactly 4 strategies, got {len(strategies)}")

    strategy_weights = None
    if args.strategy_weights:
        strategy_weights = {}
        for pair in args.strategy_weights.split(","):
            k, v = pair.split("=", 1)
            strategy_weights[k.strip()] = v.strip()

    stats = run_tournament(strategies, args.game_samples, args.weights, strategy_weights)
    report(strategies, stats)


if __name__ == "__main__":
    main()
