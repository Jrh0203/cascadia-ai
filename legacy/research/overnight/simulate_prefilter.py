#!/usr/bin/env python3
"""Simulate alternative prefilter strategies on the raw rank-correlation data.

Each row in the JSONL represents one candidate-at-a-position with its
NNUE rank/score and MCE rank/mean/std. We can evaluate any ranking rule
that uses only those fields (or board-state fields we already have) and
measure its K=8 hit rate offline — no Rust rebuild needed.

This lets us iterate on prefilter designs in seconds rather than minutes.

Strategies evaluated:
  v0 — baseline NNUE score
  v1 — diversity quota per wildlife type
  v2 — v1 + UCB-style upper-confidence selection (requires MCE priors, so only
       for simulation purposes — we'll add a cheap approximation in Rust)
  v3 — v1 + NNUE-eval-bonus for underrepresented wildlife types
  v4 — "wildlife-aware blending": shift NNUE-ranking within wildlife buckets
  v5 — per-wildlife top-N then merge

Each strategy must produce a top-K ordering from the full NNUE-scored pool.
Report: K=1/4/8/12/16 hit rate per strategy.
"""
import json
import sys
from collections import defaultdict, Counter


def load_positions(path):
    """Returns dict (game, turn) -> list of candidate dicts."""
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


def eval_strategy(positions, sort_key, name, K_vals=(1, 4, 8, 12, 16, 20, 24)):
    """Given a sort_key function (cand -> sortable), report K-hit rates.

    sort_key should return a value where HIGHER is better.
    """
    hits = {k: 0 for k in K_vals}
    total = 0
    missed_wildlife = Counter()
    by_turn = defaultdict(lambda: {k: 0 for k in K_vals} | {'n': 0})

    for (g, t), cands in positions.items():
        if not cands:
            continue
        mce_best = next((c for c in cands if c['mce_rank'] == 0), None)
        if mce_best is None:
            continue
        ranked = sorted(cands, key=lambda c: (-sort_key(c, cands), c['nnue_rank']))
        new_rank = next((i for i, c in enumerate(ranked)
                         if c['nnue_rank'] == mce_best['nnue_rank']), len(ranked))
        total += 1
        by_turn[t]['n'] += 1
        for k in K_vals:
            if new_rank < k:
                hits[k] += 1
                by_turn[t][k] += 1
        if new_rank >= 8:
            missed_wildlife[mce_best['wildlife']] += 1

    print(f"--- {name} ---")
    for k in K_vals:
        print(f"  K={k:>3}: {100.0*hits[k]/max(total,1):5.1f}% "
              f"({hits[k]}/{total})")

    # per-wildlife miss at K=8
    print(f"  miss@K=8 by wildlife: ", end='')
    for wl in ['bear','elk','salmon','hawk','fox']:
        print(f"{wl}={missed_wildlife.get(wl,0):3} ", end='')
    print()

    # per-turn bucket
    buckets = [(1,5),(6,10),(11,15),(16,20)]
    k8_by_bucket = {}
    for lo, hi in buckets:
        n = sum(by_turn[t]['n'] for t in range(lo, hi+1))
        h = sum(by_turn[t][8] for t in range(lo, hi+1))
        k8_by_bucket[(lo,hi)] = 100.0*h/max(n,1)
    print(f"  K=8 by turn-bucket: " + "  ".join(
        f"{lo}-{hi}:{k8_by_bucket[(lo,hi)]:.1f}%" for lo,hi in buckets))
    print()
    return hits[8] / max(total, 1)


# ----- STRATEGIES -----

def v0_nnue_score(c, all_cands):
    return c['nnue_score']


def v1_diversity_quota(per_wl_quota=(2, 2, 2, 2, 2)):
    """Reserve quota slots per wildlife; MCE-best gets promoted within its quota.

    Simulated as: sort primarily by (is-within-quota-for-its-wildlife DESC, nnue_rank ASC).
    """
    wl_to_i = {'bear':0,'elk':1,'salmon':2,'hawk':3,'fox':4}

    def key(c, all_cands):
        # Need to know if this candidate is in the top `quota` of its wildlife class.
        if not hasattr(key, '_cache') or key._cache_for is not all_cands:
            by_wl = defaultdict(list)
            for cc in all_cands:
                by_wl[cc['wildlife']].append(cc)
            for wl, lst in by_wl.items():
                lst.sort(key=lambda x: x['nnue_rank'])
                quota = per_wl_quota[wl_to_i.get(wl, 0)]
                for i, cc in enumerate(lst):
                    cc['_quota_rank'] = i  # 0 = best of its wildlife
            key._cache = True
            key._cache_for = all_cands
        qrank = c.get('_quota_rank', 999)
        quota = per_wl_quota[wl_to_i.get(c['wildlife'], 0)]
        in_quota = qrank < quota
        # return a big bonus if in quota, then nnue_score tiebreak
        return (1e9 if in_quota else 0) - c['nnue_rank'] * 1e6 + c['nnue_score']
    return key


def v3_underrep_bonus(bonus=3.0):
    """Penalty: boost candidates whose wildlife type is underrepresented in NNUE top-8."""
    def key(c, all_cands):
        if not hasattr(key, '_cache') or key._cache_for is not all_cands:
            top8 = sorted(all_cands, key=lambda x: x['nnue_rank'])[:8]
            wl_counts = Counter(cc['wildlife'] for cc in top8)
            for cc in all_cands:
                n_this_wl = wl_counts.get(cc['wildlife'], 0)
                # bonus = 0 if wildlife has 2+ in top-8, else decreasing boost
                cc['_boost'] = max(0, 2 - n_this_wl) * bonus
            key._cache = True
            key._cache_for = all_cands
        return c['nnue_score'] + c.get('_boost', 0)
    return key


def v4_per_wildlife_then_merge(per_wl=(2, 2, 2, 2, 2)):
    """Pick top-N per wildlife class, then merge sorted by nnue_score."""
    wl_to_i = {'bear':0,'elk':1,'salmon':2,'hawk':3,'fox':4}

    def key(c, all_cands):
        if not hasattr(key, '_cache') or key._cache_for is not all_cands:
            by_wl = defaultdict(list)
            for cc in all_cands:
                by_wl[cc['wildlife']].append(cc)
            keep = set()
            for wl, lst in by_wl.items():
                lst.sort(key=lambda x: x['nnue_rank'])
                n = per_wl[wl_to_i.get(wl, 0)]
                for cc in lst[:n]:
                    keep.add(id(cc))
            for cc in all_cands:
                cc['_keep'] = id(cc) in keep
            key._cache = True
            key._cache_for = all_cands
        # huge bonus if kept, else drop to NNUE ordering
        return (1e9 if c.get('_keep', False) else 0) + c['nnue_score']
    return key


def v5_pattern_aware_quota():
    """Quota per wildlife, BUT within a wildlife class, prefer candidates that
    actually plausibly extend a pattern (e.g. for bear: place adjacent to
    existing bear; for elk: extend a line). We approximate using board state
    + wildlife counts from the JSONL (which has wl_bear/wl_elk/... at turn start).
    """
    wl_to_i = {'bear':0,'elk':1,'salmon':2,'hawk':3,'fox':4}

    def quota_for(c):
        # Dynamic quota based on board state
        wl_count = {
            'bear': c['wl_bear'], 'elk': c['wl_elk'],
            'salmon': c['wl_salmon'], 'hawk': c['wl_hawk'], 'fox': c['wl_fox'],
        }
        tl = c['turns_left']
        q = {'bear':2, 'elk':2, 'salmon':2, 'hawk':2, 'fox':2}
        # Reduce fox quota when fox already placed a few
        if wl_count['fox'] >= 3: q['fox'] = 1
        # Increase bear quota if working on pairs (odd count suggests incomplete)
        if wl_count['bear'] % 2 == 1: q['bear'] = 3
        # Increase hawk if not yet at 5+
        if wl_count['hawk'] < 5 and tl > 20: q['hawk'] = 3
        # Increase elk if any elk placed (line-building in progress)
        if 1 <= wl_count['elk'] <= 3 and tl > 20: q['elk'] = 3
        # Increase salmon similarly
        if 1 <= wl_count['salmon'] <= 5 and tl > 20: q['salmon'] = 3
        return q

    def key(c, all_cands):
        if not hasattr(key, '_cache') or key._cache_for is not all_cands:
            # Use first candidate to read board state
            first = all_cands[0]
            q = quota_for(first)
            by_wl = defaultdict(list)
            for cc in all_cands:
                by_wl[cc['wildlife']].append(cc)
            keep = set()
            for wl, lst in by_wl.items():
                lst.sort(key=lambda x: x['nnue_rank'])
                n = q.get(wl, 2)
                for cc in lst[:n]:
                    keep.add(id(cc))
            for cc in all_cands:
                cc['_keep'] = id(cc) in keep
            key._cache = True
            key._cache_for = all_cands
        return (1e9 if c.get('_keep', False) else 0) + c['nnue_score']
    return key


def v6_per_wl_plus_cv():
    """v4 (per-wildlife) with MCE-score-like re-rank using NNUE priors +
    a 'control variate' style variance penalty. Since we don't have cheap MCE
    here, approximate with nnue_score only — but within class, keep top-N."""
    return v4_per_wildlife_then_merge(per_wl=(3, 3, 3, 3, 2))


def v7_wide_per_wildlife():
    """How well do we do if we just keep top-4 of each wildlife? K=20 in practice."""
    return v4_per_wildlife_then_merge(per_wl=(4, 4, 4, 4, 4))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'overnight/rank_corr_raw_50g_100r.jsonl'
    print(f"Loading {path}...")
    positions = load_positions(path)
    print(f"Loaded {len(positions)} positions\n")

    strategies = [
        ('v0 NNUE baseline', v0_nnue_score),
        ('v1 diversity quota=2 each', v1_diversity_quota((2,2,2,2,2))),
        ('v1b diversity quota=3bear 2elk/salmon/hawk 1fox', v1_diversity_quota((3,2,2,2,1))),
        ('v3 underrep bonus=3', v3_underrep_bonus(3.0)),
        ('v3b underrep bonus=5', v3_underrep_bonus(5.0)),
        ('v4 per-wildlife top-2', v4_per_wildlife_then_merge((2,2,2,2,2))),
        ('v4b per-wildlife top-2,2,2,2,1 + 1 extra', v4_per_wildlife_then_merge((2,2,2,2,1))),
        ('v5 dynamic pattern quota', v5_pattern_aware_quota()),
        ('v6 per-wildlife top-3', v4_per_wildlife_then_merge((3,3,3,3,2))),
        ('v7 per-wildlife top-4', v4_per_wildlife_then_merge((4,4,4,4,4))),
    ]

    results = []
    for name, key_fn in strategies:
        # Reset any cached state
        if hasattr(key_fn, '_cache'):
            delattr(key_fn, '_cache')
            delattr(key_fn, '_cache_for')
        h = eval_strategy(positions, key_fn, name)
        results.append((name, h))

    print("\n\n=== SUMMARY (K=8 hit rate) ===")
    results.sort(key=lambda x: -x[1])
    for name, h in results:
        print(f"  {h*100:5.1f}%  {name}")


if __name__ == '__main__':
    main()
