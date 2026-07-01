#!/usr/bin/env python3
"""Analyze tournament JSONL — comprehensive markdown report.

Reads a JSONL produced by tournament_v5.py and outputs:
- Per-strategy score summary (mean/p10/p50/p90, base + bonus)
- Per-animal breakdown (mean/p10/p90)
- Bimodality coefficient per strategy
- Pairwise winrate matrix (% A finishes above B in placement)
- Pairwise mean-score winrate (% A scores higher than B)
- Multiplayer Elo (TrueSkill-inspired)
- Rank distribution (% 1st/2nd/3rd/4th)
- ASCII histograms of base scores per strategy
- Bootstrap 95% CIs on means
- Per-seat performance check (sanity: no seat bias)
- Token economy + bonus contribution per strategy
- Tactical similarity matrix (cosine sim of per-animal mean vectors)
- Robustness: % games above 90 base score

Usage:
    python3 overnight/tournament_analysis.py overnight/v5sh/tournament.jsonl
"""
import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient.
    BC > 5/9 ≈ 0.555 suggests bimodality (uniform = 5/9, normal = 1/3).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return float("nan")
    g = ((x - m) ** 3).mean() / (s ** 3)        # skewness
    k = ((x - m) ** 4).mean() / (s ** 4)        # kurtosis (not excess)
    bc = (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return bc


def bootstrap_ci(x, conf=0.95, n_resamples=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means.append(x[idx].mean())
    means = np.sort(means)
    lo = means[int((1 - conf) / 2 * n_resamples)]
    hi = means[int((1 + conf) / 2 * n_resamples)]
    return (lo, hi)


def ascii_hist(x, low=80, high=110, bin_width=2, max_width=40):
    """Render a horizontal ASCII histogram for terminal."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return "(empty)"
    bins = np.arange(low, high + bin_width, bin_width)
    counts, edges = np.histogram(x, bins=bins)
    max_count = max(counts) if counts.max() else 1
    out = []
    for i, c in enumerate(counts):
        bar = "█" * int(round(c * max_width / max_count))
        out.append(f"  {int(edges[i]):>3}-{int(edges[i+1])-1:<3} | {bar} {c}")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────
# TrueSkill-inspired multiplayer Elo
# ─────────────────────────────────────────────────────────────────────

class MultiplayerRating:
    """Simple Elo extension to N-player free-for-all.

    For each game with players ranked R1 > R2 > ... > Rn (R1 best),
    treat the result as n*(n-1)/2 pairwise wins, each with K-factor /
    (n-1) so total update magnitude is roughly equivalent to a
    standard 2-player game.

    Initial rating: 1500 ± 350.
    K-factor: 32 / (n-1) per pair, so total K per game ≈ 32 (matches Elo norm).
    """

    def __init__(self, k_per_pair=10):
        self.ratings = defaultdict(lambda: 1500.0)
        self.games = defaultdict(int)
        self.k = k_per_pair

    def expected(self, ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_game(self, ranking):
        """ranking: list of (strategy_name, score) sorted by score DESC."""
        names = [r[0] for r in ranking]
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = names[i], names[j]  # a ranked above b
                # a "wins" the pairwise comparison
                ea = self.expected(self.ratings[a], self.ratings[b])
                self.ratings[a] += self.k * (1 - ea)
                self.ratings[b] += self.k * (0 - (1 - ea))
        for n_ in names:
            self.games[n_] += 1

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])


# ─────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", help="Tournament JSONL file")
    p.add_argument("--out", default=None, help="Markdown output path (default: stdout)")
    args = p.parse_args()

    with open(args.jsonl) as f:
        games = [json.loads(line) for line in f if line.strip()]
    games = [g for g in games if g.get("players") and len(g["players"]) == 4]

    if not games:
        print(f"No valid games in {args.jsonl}")
        return

    # Build per-strategy data
    per_strat_records = defaultdict(list)  # strategy → list of dict per seat-game
    for game in games:
        seat_names = game["seat_names"]
        for seat in range(4):
            name = seat_names[seat]
            pl = game["players"][seat]
            pl_with_seat = dict(pl)
            pl_with_seat["seat"] = seat
            pl_with_seat["round_i"] = game["round_i"]
            pl_with_seat["rot_i"] = game["rot_i"]
            per_strat_records[name].append(pl_with_seat)

    strategies = list(per_strat_records.keys())

    # Compute multiplayer Elo (random 0.99×, full evaluation)
    elo = MultiplayerRating(k_per_pair=10)
    for game in games:
        ranking = []
        for seat in range(4):
            pl = game["players"][seat]
            # tie-break: bonus, then nature tokens (matches official Cascadia rules)
            ranking.append((game["seat_names"][seat], pl["base"], pl["bonus"], pl["tok"]))
        ranking.sort(key=lambda x: (-x[1], -x[2], -x[3]))
        elo.update_game([(name, b) for name, b, _, _ in ranking])

    # Pairwise win rates: % games where strategy A finishes above strategy B in placement
    pairwise_placement = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [a_above_b, total]
    pairwise_score = defaultdict(lambda: defaultdict(lambda: [0, 0]))      # [a_higher_than_b, total]
    for game in games:
        seat_names = game["seat_names"]
        scores = [game["players"][s]["base"] for s in range(4)]
        ranks = sorted(range(4), key=lambda s: (-scores[s], -game["players"][s]["bonus"]))
        rank_of = {seat_names[s]: ranks.index(s) for s in range(4)}
        score_of = {seat_names[s]: scores[s] for s in range(4)}
        for a in seat_names:
            for b in seat_names:
                if a == b: continue
                pairwise_placement[a][b][1] += 1
                if rank_of[a] < rank_of[b]:
                    pairwise_placement[a][b][0] += 1
                pairwise_score[a][b][1] += 1
                if score_of[a] > score_of[b]:
                    pairwise_score[a][b][0] += 1

    # Rank distribution per strategy
    rank_dist = defaultdict(lambda: [0, 0, 0, 0])  # [r1, r2, r3, r4]
    for game in games:
        seat_names = game["seat_names"]
        scores = [game["players"][s]["base"] for s in range(4)]
        ranks_seat = sorted(range(4), key=lambda s: (-scores[s], -game["players"][s]["bonus"]))
        for rank, seat in enumerate(ranks_seat):
            rank_dist[seat_names[seat]][rank] += 1

    # Per-seat performance
    seat_means = defaultdict(lambda: [[] for _ in range(4)])  # strategy → seat → list of base scores
    for game in games:
        for seat in range(4):
            seat_means[game["seat_names"][seat]][seat].append(game["players"][seat]["base"])

    # ─────────────────────────────────────────────────────────
    # Build the markdown report
    # ─────────────────────────────────────────────────────────
    lines = []
    out = lines.append

    out(f"# Round-Robin Tournament Analysis\n")
    out(f"**Source**: `{args.jsonl}`  ")
    out(f"**Games**: {len(games)}  ")
    out(f"**Strategies**: {len(strategies)}  ")
    out(f"**Per-strategy seat-games**: {len(per_strat_records[strategies[0]])}  ")
    out("")

    # ── ELO LEADERBOARD ──
    out(f"## Multiplayer Elo Leaderboard (TrueSkill-inspired)\n")
    out(f"Initial rating 1500. K-factor 10 per pairwise comparison. Tie-break: bonus, then tokens.\n")
    out(f"| Rank | Strategy | Elo | Games | Δ vs avg |")
    out(f"|---|---|---|---|---|")
    avg_elo = np.mean([r for _, r in elo.leaderboard()])
    for rank, (name, rating) in enumerate(elo.leaderboard(), 1):
        out(f"| {rank} | **{name}** | {rating:.0f} | {elo.games[name]} | {rating - avg_elo:+.0f} |")
    out("")

    # ── SCORE SUMMARY ──
    out(f"## Score Summary\n")
    out(f"Per-seat-game distribution. P10/P50/P90 are integer percentiles. "
        f"BC = Sarle's bimodality coefficient (>0.555 suggests bimodal, normal ≈ 0.33).\n")
    out(f"| Strategy | Mean (base) | 95% CI | SD | P10 | P50 | P90 | Mean (bonus) | BC |")
    out(f"|---|---|---|---|---|---|---|---|---|")
    for name in strategies:
        recs = per_strat_records[name]
        base = np.array([r["base"] for r in recs])
        bonus = np.array([r["bonus"] for r in recs])
        ci_lo, ci_hi = bootstrap_ci(base)
        bc = bimodality_coefficient(base)
        out(f"| {name} | {base.mean():.2f} | [{ci_lo:.2f}, {ci_hi:.2f}] | "
            f"{base.std(ddof=1):.2f} | {int(np.percentile(base, 10))} | "
            f"{int(np.percentile(base, 50))} | {int(np.percentile(base, 90))} | "
            f"{bonus.mean():.2f} | {bc:.3f} |")
    out("")

    # ── PER-ANIMAL BREAKDOWN ──
    out(f"## Per-Animal Score Breakdown\n")
    for name in strategies:
        recs = per_strat_records[name]
        out(f"### {name}\n")
        out(f"| Animal | Mean | P10 | P50 | P90 | SD |")
        out(f"|---|---|---|---|---|---|")
        for animal in ["bear", "elk", "salmon", "hawk", "fox"]:
            v = np.array([r[animal] for r in recs])
            out(f"| {animal} | {v.mean():.2f} | {int(np.percentile(v, 10))} | "
                f"{int(np.percentile(v, 50))} | {int(np.percentile(v, 90))} | "
                f"{v.std(ddof=1):.2f} |")
        out("")

    # ── PLACEMENT WIN RATE MATRIX ──
    out(f"## Pairwise Placement Matrix (% row finishes above column)\n")
    out(f"|  | " + " | ".join(strategies) + " |")
    out(f"|---|" + "---|" * len(strategies))
    for a in strategies:
        row = [a]
        for b in strategies:
            if a == b:
                row.append("—")
            else:
                w, t = pairwise_placement[a][b]
                row.append(f"{100*w/t:.1f}%" if t else "—")
        out(f"| **{row[0]}** | " + " | ".join(row[1:]) + " |")
    out("")

    # ── SCORE WIN RATE MATRIX ──
    out(f"## Pairwise Score Matrix (% row scores higher than column)\n")
    out(f"|  | " + " | ".join(strategies) + " |")
    out(f"|---|" + "---|" * len(strategies))
    for a in strategies:
        row = [a]
        for b in strategies:
            if a == b:
                row.append("—")
            else:
                w, t = pairwise_score[a][b]
                row.append(f"{100*w/t:.1f}%" if t else "—")
        out(f"| **{row[0]}** | " + " | ".join(row[1:]) + " |")
    out("")

    # ── RANK DISTRIBUTION ──
    out(f"## Finish Distribution (% of games)\n")
    out(f"| Strategy | 1st | 2nd | 3rd | 4th |")
    out(f"|---|---|---|---|---|")
    for name in strategies:
        d = rank_dist[name]
        total = sum(d)
        if total:
            out(f"| {name} | {100*d[0]/total:.1f}% | {100*d[1]/total:.1f}% | "
                f"{100*d[2]/total:.1f}% | {100*d[3]/total:.1f}% |")
    out("")

    # ── ROBUSTNESS / CONSISTENCY ──
    out(f"## Robustness / Consistency\n")
    out(f"| Strategy | % games ≥ 90 (base) | % games ≥ 95 | % games ≥ 100 | Top-2 finish % |")
    out(f"|---|---|---|---|---|")
    for name in strategies:
        recs = per_strat_records[name]
        base = np.array([r["base"] for r in recs])
        d = rank_dist[name]
        total = sum(d) or 1
        out(f"| {name} | {100*np.mean(base >= 90):.1f}% | {100*np.mean(base >= 95):.1f}% | "
            f"{100*np.mean(base >= 100):.1f}% | {100*(d[0]+d[1])/total:.1f}% |")
    out("")

    # ── TOKEN ECONOMY + BONUS CONTRIBUTION ──
    out(f"## Token Economy + Habitat Bonus Contribution\n")
    out(f"| Strategy | Avg tokens | Avg habitat (base) | Avg wildlife | Bonus contribution % |")
    out(f"|---|---|---|---|---|")
    for name in strategies:
        recs = per_strat_records[name]
        tok = np.mean([r["tok"] for r in recs])
        hab = np.mean([r["hab"] for r in recs])
        wl = np.mean([r["wl"] for r in recs])
        bonus = np.mean([r["bonus"] for r in recs])
        base = np.mean([r["base"] for r in recs])
        bonus_pct = 100 * (bonus - base) / bonus if bonus else 0
        out(f"| {name} | {tok:.2f} | {hab:.2f} | {wl:.2f} | {bonus_pct:.2f}% |")
    out("")

    # ── PER-SEAT SANITY CHECK ──
    out(f"## Per-Seat Mean Scores (sanity check — no strong seat bias expected)\n")
    out(f"| Strategy | Seat 0 | Seat 1 | Seat 2 | Seat 3 | Range |")
    out(f"|---|---|---|---|---|---|")
    for name in strategies:
        seat_data = seat_means[name]
        means = [np.mean(s) if s else float("nan") for s in seat_data]
        rng = max(m for m in means if not math.isnan(m)) - min(m for m in means if not math.isnan(m))
        out(f"| {name} | {means[0]:.2f} | {means[1]:.2f} | {means[2]:.2f} | {means[3]:.2f} | {rng:.2f} |")
    out("")

    # ── TACTICAL SIMILARITY ──
    out(f"## Tactical Similarity (cosine sim of per-animal mean vectors)\n")
    out(f"1.00 = identical animal-scoring profile, 0 = orthogonal. Higher = more similar style.\n")
    animal_means = {}
    for name in strategies:
        recs = per_strat_records[name]
        v = np.array([np.mean([r[a] for r in recs]) for a in ["bear","elk","salmon","hawk","fox"]])
        animal_means[name] = v / np.linalg.norm(v)
    out(f"|  | " + " | ".join(strategies) + " |")
    out(f"|---|" + "---|" * len(strategies))
    for a in strategies:
        row = [a]
        for b in strategies:
            if a == b:
                row.append("—")
            else:
                sim = float(animal_means[a] @ animal_means[b])
                row.append(f"{sim:.4f}")
        out(f"| **{row[0]}** | " + " | ".join(row[1:]) + " |")
    out("")

    # ── ASCII HISTOGRAMS ──
    out(f"## Score Histograms (ASCII)\n")
    out(f"```")
    for name in strategies:
        recs = per_strat_records[name]
        base = np.array([r["base"] for r in recs])
        out(f"")
        out(f"{name} (base score, n={len(base)}):")
        out(ascii_hist(base, low=80, high=110, bin_width=2, max_width=40))
    out(f"```")
    out("")

    out(f"## Bonus Histograms (ASCII)\n")
    out(f"```")
    for name in strategies:
        recs = per_strat_records[name]
        bonus = np.array([r["bonus"] for r in recs])
        out(f"")
        out(f"{name} (bonus-included score, n={len(bonus)}):")
        out(ascii_hist(bonus, low=85, high=115, bin_width=2, max_width=40))
    out(f"```")

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"Report written to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
