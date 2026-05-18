#!/usr/bin/env python3
"""Simulate a 'micro-rollout' prefilter: cheap MCE with few rollouts.

Idea: do a tiny MCE pass (R_prefilter rollouts per candidate) at the
prefilter stage, then keep top-K by micro-rollout mean. Since the raw
JSONL already has mean/std/min/max per candidate (from 100 rollouts),
we can simulate ANY rollout count by drawing from the gaussian
approximation N(mean, std).

This lets us answer: how many micro-rollouts (R) on how many candidates
(N) do we need to reliably catch MCE-best in top-8?

Strategies:
  - "naive": R rollouts on ALL candidates, pick top-8
  - "prefiltered": NNUE pre-rank, R rollouts on top-N by NNUE, pick top-8
  - "hybrid": diversity quota first, then R rollouts within quota
"""
import json
import sys
import random
from collections import defaultdict, Counter
import math


def load_positions(path):
    pos = defaultdict(list)
    for raw in open(path):
        raw = raw.strip()
        if not raw.startswith('{'):
            continue
        obj = json.loads(raw)
        if obj.get('type') == 'score':
            continue
        if 'nnue_rank' not in obj:
            continue
        pos[(obj['game'], obj['turn'])].append(obj)
    return pos


def draw_mean(c, R, rng):
    """Draw R samples from approx N(mean, std), return simulated mean."""
    if R <= 0:
        return c['nnue_score']
    mean = c['mce_mean']
    std = c['mce_std']
    if std == 0 or R >= 100:
        return mean
    # approximate by gaussian. std of mean estimator = std / sqrt(R)
    return rng.gauss(mean, std / math.sqrt(R))


def eval_strategy(positions, strategy, name, K=8, n_trials=5):
    """Average across trials (each trial draws fresh micro-rollout noise)."""
    hits = 0
    total = 0
    all_per_trial = []

    for trial in range(n_trials):
        rng = random.Random(42 + trial * 17)
        hit_this = 0
        total_this = 0
        for (g, t), cands in positions.items():
            mce_best = next((c for c in cands if c['mce_rank'] == 0), None)
            if mce_best is None:
                continue
            total_this += 1
            ranked = strategy(cands, rng)
            if any(c['nnue_rank'] == mce_best['nnue_rank']
                   for c in ranked[:K]):
                hit_this += 1
        all_per_trial.append(hit_this / max(total_this, 1))
        hits += hit_this
        total += total_this
    mean_hit = hits / max(total, 1)
    trial_std = 0 if n_trials < 2 else (
        sum((h - mean_hit)**2 for h in all_per_trial) / (n_trials - 1)
    ) ** 0.5
    print(f"  {name:<45}: K={K} hit = {100*mean_hit:5.1f}% ± {100*trial_std:.1f}%  "
          f"({hits}/{total})")
    return mean_hit


# ----- STRATEGIES -----

def strat_baseline_nnue(cands, rng):
    return sorted(cands, key=lambda c: c['nnue_rank'])


def strat_naive_full_mce(R):
    def f(cands, rng):
        scored = [(c, draw_mean(c, R, rng)) for c in cands]
        scored.sort(key=lambda x: -x[1])
        return [c for c, _ in scored]
    return f


def strat_prefiltered(N, R):
    """Take NNUE top-N, run R micro-rollouts, pick best."""
    def f(cands, rng):
        top_n = sorted(cands, key=lambda c: c['nnue_rank'])[:N]
        scored = [(c, draw_mean(c, R, rng)) for c in top_n]
        scored.sort(key=lambda x: -x[1])
        return [c for c, _ in scored]
    return f


def strat_diverse_then_rollout(per_wl, R):
    """Per-wildlife top-N by NNUE, then R rollouts, pick top-8 by rollout mean."""
    wl_to_i = {'bear':0,'elk':1,'salmon':2,'hawk':3,'fox':4}
    def f(cands, rng):
        by_wl = defaultdict(list)
        for c in cands:
            by_wl[c['wildlife']].append(c)
        pool = []
        for wl, lst in by_wl.items():
            lst.sort(key=lambda x: x['nnue_rank'])
            n = per_wl[wl_to_i.get(wl, 0)]
            pool.extend(lst[:n])
        scored = [(c, draw_mean(c, R, rng)) for c in pool]
        scored.sort(key=lambda x: -x[1])
        return [c for c, _ in scored]
    return f


def strat_two_phase(N_nnue, per_wl_extra, R):
    """Phase 1: NNUE top-N. Phase 2: per-wildlife top-K from REMAINING.
    Then R rollouts on combined pool."""
    wl_to_i = {'bear':0,'elk':1,'salmon':2,'hawk':3,'fox':4}
    def f(cands, rng):
        by_nnue = sorted(cands, key=lambda c: c['nnue_rank'])
        phase1 = by_nnue[:N_nnue]
        phase1_ids = set(id(c) for c in phase1)
        by_wl = defaultdict(list)
        for c in cands:
            if id(c) not in phase1_ids:
                by_wl[c['wildlife']].append(c)
        extras = []
        for wl, lst in by_wl.items():
            lst.sort(key=lambda x: x['nnue_rank'])
            n = per_wl_extra[wl_to_i.get(wl, 0)]
            extras.extend(lst[:n])
        pool = phase1 + extras
        scored = [(c, draw_mean(c, R, rng)) for c in pool]
        scored.sort(key=lambda x: -x[1])
        return [c for c, _ in scored]
    return f


def strat_halving(N, R_total):
    """Sequential halving: start with N, R_total rollouts split across rounds.
    Round 0: N cands, R rollouts each. Drop bottom half.
    Round 1: N/2 cands, R more rollouts each. ...
    Final: top-8."""
    def f(cands, rng):
        pool = sorted(cands, key=lambda c: c['nnue_rank'])[:N]
        # collect rollouts cumulatively
        cum_scores = {id(c): [] for c in pool}
        rounds = max(1, int(math.log2(max(1, len(pool) // 8))))
        per_round = max(1, R_total // rounds)
        while len(pool) > 8:
            for c in pool:
                cum_scores[id(c)].append(draw_mean(c, per_round, rng))
            means = [(c, sum(cum_scores[id(c)]) / len(cum_scores[id(c)])) for c in pool]
            means.sort(key=lambda x: -x[1])
            half = max(8, len(pool) // 2)
            pool = [c for c, _ in means[:half]]
        # final rank
        means = [(c, sum(cum_scores[id(c)]) / max(1, len(cum_scores[id(c)]))) for c in pool]
        means.sort(key=lambda x: -x[1])
        return [c for c, _ in means]
    return f


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'overnight/rank_corr_raw_50g_100r.jsonl'
    print(f"Loading {path}...")
    positions = load_positions(path)
    print(f"Loaded {len(positions)} positions\n")
    print(f"Target: K=8 hit >= 85%\n")
    print("=" * 70)
    print("Strategy                                              K=8 hit        rollout budget")
    print("=" * 70)

    eval_strategy(positions, strat_baseline_nnue, "v0 NNUE baseline", K=8)

    print("\n-- Naive: R rollouts on ALL candidates --")
    for R in [2, 5, 10, 20, 50]:
        eval_strategy(positions, strat_naive_full_mce(R),
                      f"naive R={R} on all ~80", K=8)

    print("\n-- NNUE-top-N + R micro-rollouts --")
    for (N, R) in [(16,5),(16,10),(20,5),(20,10),(20,20),(24,5),(24,10),(32,5),(32,10)]:
        eval_strategy(positions, strat_prefiltered(N, R),
                      f"NNUE top-{N} + {R} rollouts (budget={N*R})", K=8)

    print("\n-- Per-wildlife + R micro-rollouts --")
    for per, R in [((3,3,3,3,2), 10), ((4,4,4,4,2), 10), ((4,4,4,4,4), 10),
                    ((3,3,3,3,2), 20), ((4,4,4,4,4), 20)]:
        n_total = sum(per)
        eval_strategy(positions, strat_diverse_then_rollout(per, R),
                      f"per-wl {per} + {R} rollouts (budget≈{n_total*R})", K=8)

    print("\n-- Two-phase (NNUE-top-M + per-wildlife extras) + R rollouts --")
    for (N, extra, R) in [(10,(1,1,1,1,0),10), (10,(2,2,2,2,1),10),
                           (12,(2,2,2,2,1),10), (8,(2,2,2,2,1),15)]:
        pool = N + sum(extra)
        eval_strategy(positions, strat_two_phase(N, extra, R),
                      f"NNUE top-{N} + per-wl extras {extra} + {R} rollouts (budget≈{pool*R})", K=8)

    print("\n-- Sequential halving --")
    for N, R in [(20, 200), (24, 200), (32, 300), (20, 100), (16, 100)]:
        eval_strategy(positions, strat_halving(N, R),
                      f"halving N={N} R_total={R}", K=8)


if __name__ == '__main__':
    main()
