"""Symmetric-play experiment: all 4 players use the same strategy.

Measures:
- Per-player score distribution (mean, median, p10, p90)
- Wildlife distribution per player (bear/elk/salmon/hawk/fox)
- Total bear consumption across all 4 players (vs 20 in the bag)
- Score variance within each game (winner-loser gap)

Uses the CLI with CASCADIA_OPPONENTS_SAME=1, which makes opponents run the
same strategy as player 0 AND dumps per-player stats to stderr.

Usage:
    python3 overnight/symmetric_experiment.py --games 20 --rollouts 100
"""

import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
import numpy as np


def run_symmetric_bench(games, rollouts, weights):
    """Run the CLI with opponents-same flag, return list of per-game per-player dicts."""
    cmd = [
        "./target/release/cascadia-cli", str(games),
        "--nnue-rollout-mce",
        "--rollouts", str(rollouts),
        "--alloc", "halving",
        "--candidates", "expanded",
        "--prefilter-k", "8",
        "--weights", weights,
    ]
    env = os.environ.copy()
    env["CASCADIA_OPPONENTS_SAME"] = "1"
    print(f"Running: {' '.join(cmd)}")
    print(f"  CASCADIA_OPPONENTS_SAME=1")
    print()

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    # Parse stderr for per-player records
    games_data = []
    current_game = None
    player_re = re.compile(
        r"SYMPLAYER p=(\d+) base=(\d+) bonus=(\d+) hab=(\d+) wl=(\d+) tok=(\d+) "
        r"bear=(\d+) elk=(\d+) salmon=(\d+) hawk=(\d+) fox=(\d+)"
    )

    game_done_count = 0
    for line_raw in proc.stderr:
        line = line_raw.decode(errors="replace").rstrip()
        if "SYMMETRIC_GAME_BEGIN" in line:
            current_game = []
        elif "SYMMETRIC_GAME_END" in line:
            if current_game is not None:
                games_data.append(current_game)
                game_done_count += 1
                elapsed = time.time() - t0
                rate = game_done_count / elapsed
                eta = (games - game_done_count) / max(rate, 0.001)
                print(f"  Game {game_done_count}/{games} done ({elapsed:.0f}s elapsed, "
                      f"ETA {eta:.0f}s)")
            current_game = None
        elif current_game is not None:
            m = player_re.search(line)
            if m:
                current_game.append({
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

    proc.stdout.read()
    proc.wait()
    return games_data


def analyze(games_data):
    """Print aggregated stats across all games/players."""
    print("\n" + "=" * 78)
    print(f"SYMMETRIC EXPERIMENT RESULTS ({len(games_data)} games × 4 players = "
          f"{len(games_data) * 4} player-samples)")
    print("=" * 78)

    if not games_data:
        print("No games data recorded.")
        return

    # Flatten: all player records from all games
    all_players = [p for g in games_data for p in g]

    def dist(key):
        return np.array([p[key] for p in all_players])

    print("\n--- Score Distribution (all player-seats combined) ---")
    for key in ["base", "bonus", "hab", "wl", "tok"]:
        v = dist(key)
        print(f"  {key:>6}  mean={v.mean():5.2f}  med={np.median(v):5.1f}  "
              f"p10={np.percentile(v, 10):5.1f}  p90={np.percentile(v, 90):5.1f}  "
              f"min/max={v.min()}/{v.max()}")

    print("\n--- Per-Animal Score Distribution ---")
    for key in ["bear", "elk", "salmon", "hawk", "fox"]:
        v = dist(key)
        print(f"  {key:>6}  mean={v.mean():5.2f}  med={np.median(v):5.1f}  "
              f"p10={np.percentile(v, 10):5.1f}  p90={np.percentile(v, 90):5.1f}  "
              f"min/max={v.min()}/{v.max()}")

    # Within-game stats: winner-loser gap, rank distribution
    print("\n--- Within-Game Competition ---")
    winner_scores = []
    loser_scores = []
    gaps = []
    rank_counts = defaultdict(int)
    for g in games_data:
        bases = [p["base"] for p in g]
        mx, mn = max(bases), min(bases)
        winner_scores.append(mx)
        loser_scores.append(mn)
        gaps.append(mx - mn)
        # Rank each player (0=best)
        sorted_idx = sorted(range(len(g)), key=lambda i: -g[i]["base"])
        for rank, idx in enumerate(sorted_idx):
            rank_counts[(g[idx]["p"], rank)] += 1

    print(f"  Winner mean:  {np.mean(winner_scores):.2f}  (max={max(winner_scores)})")
    print(f"  Loser mean:   {np.mean(loser_scores):.2f}  (min={min(loser_scores)})")
    print(f"  Gap mean:     {np.mean(gaps):.2f}  (max gap={max(gaps)})")

    # Bear consumption: how much bear wildlife went to all 4 players?
    # Assumes 1-pair=4, 2=11, 3=19, 4+=27 (Card A)
    print("\n--- Bear Resource Analysis (Card A: pairs 1/2/3/4+ = 4/11/19/27) ---")
    bear_totals_per_game = []
    for g in games_data:
        # Back-solve pairs from score (assumes max 4 pairs per player × 20 bears / 2 = 10 pairs in bag)
        pairs_per_player = []
        for p in g:
            b = p["bear"]
            if b == 0: pairs_per_player.append(0)
            elif b <= 4: pairs_per_player.append(1)
            elif b <= 11: pairs_per_player.append(2)
            elif b <= 19: pairs_per_player.append(3)
            else: pairs_per_player.append(4)  # 27+
        total_pairs = sum(pairs_per_player)
        bears_consumed = total_pairs * 2
        bear_totals_per_game.append((bears_consumed, pairs_per_player))

    total_bears_consumed = [bt for bt, _ in bear_totals_per_game]
    pairs_distributions = [pd for _, pd in bear_totals_per_game]
    print(f"  Bears consumed per game (mean): {np.mean(total_bears_consumed):.1f} / 20 available")
    print(f"  Pairs per game (mean per player):")
    pairs_flat = [p for game in pairs_distributions for p in game]
    pairs_hist = np.bincount(pairs_flat, minlength=5)
    for i, c in enumerate(pairs_hist):
        pct = 100 * c / len(pairs_flat)
        label = f"{i}+" if i == 4 else str(i)
        print(f"    {label} pairs: {c:3d} players ({pct:.1f}%)")

    # Strategy imprint: what % of player-samples got max bear (≥4 pairs)?
    pct_max_bear = 100 * sum(1 for p in pairs_flat if p >= 4) / len(pairs_flat)
    pct_zero_bear = 100 * sum(1 for p in pairs_flat if p == 0) / len(pairs_flat)
    print(f"  {pct_max_bear:.1f}% of player-seats achieved 4+ pairs (bear ceiling)")
    print(f"  {pct_zero_bear:.1f}% of player-seats got zero bear pairs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--rollouts", type=int, default=100)
    p.add_argument("--weights", default="nnue_weights_v9_iter14.bin")
    args = p.parse_args()

    games_data = run_symmetric_bench(args.games, args.rollouts, args.weights)
    analyze(games_data)


if __name__ == "__main__":
    main()
