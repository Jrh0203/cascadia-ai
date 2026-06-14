#!/usr/bin/env python3
"""Deep baseline analysis of the prefilter leak.

Goal: figure out WHERE the NNUE prefilter misses MCE-best moves so
the replacement has concrete targets.

Breakdowns:
  - Overall K=1..32 hit rate
  - Per-wildlife (what wildlife is MCE picking that NNUE drops?)
  - Per-turn bucket
  - Pattern context (are misses tied to building bear pairs / elk lines / salmon runs / hawk isolation?)
  - What rank does the MCE-best actually sit at? (distribution)
  - Score gap when missed (is it painful or cosmetic?)

Run:
  python3 overnight/analyze_baseline.py overnight/rank_corr_raw_50g_100r.jsonl
"""
import json
import sys
from collections import defaultdict, Counter


def load_lines(path):
    lines, scores = [], []
    for raw in open(path):
        raw = raw.strip()
        if not raw.startswith('{'):
            continue
        obj = json.loads(raw)
        if obj.get('type') == 'score':
            scores.append(obj)
        elif 'nnue_rank' in obj:
            lines.append(obj)
    return lines, scores


def mean(xs): return sum(xs) / len(xs) if xs else 0.0
def pct(x): return f"{100*x:5.1f}%"


def main(path):
    lines, scores = load_lines(path)
    print(f"Loaded {len(lines)} candidate lines, {len(scores)} score lines from {path}\n")

    # Group by (game, turn)
    pos = defaultdict(list)
    for l in lines:
        pos[(l['game'], l['turn'])].append(l)

    n_pos = len(pos)
    print(f"{n_pos} positions\n")

    # ---------- overall K-hit-rate ----------
    K_vals = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    hit = {k: 0 for k in K_vals}
    miss_details = []  # (game, turn, mce_rank, mce_mean - nnue_top1_mce_mean)
    rank_distribution = Counter()

    for key, cands in pos.items():
        cands_by_nnue = sorted(cands, key=lambda c: c['nnue_rank'])
        # MCE-best = rank_by_mce == 0
        mce_best = next((c for c in cands if c['mce_rank'] == 0), None)
        if mce_best is None:
            continue
        r = mce_best['nnue_rank']
        rank_distribution[r] += 1
        for k in K_vals:
            if r < k:
                hit[k] += 1

        nnue_top1 = cands_by_nnue[0]
        gap = mce_best['mce_mean'] - nnue_top1['mce_mean']
        miss_details.append({
            'game': key[0], 'turn': key[1],
            'mce_rank_in_nnue': r,
            'mce_best_wildlife': mce_best['wildlife'],
            'nnue_top1_wildlife': nnue_top1['wildlife'],
            'gap': gap,
            'n_cands': mce_best['n_cands'],
            'hab': mce_best.get('hab', 0),
            'tokens': mce_best.get('tokens', 0),
            'turns_left': mce_best.get('turns_left', 0),
            'wl_bear': mce_best.get('wl_bear', 0),
            'wl_elk': mce_best.get('wl_elk', 0),
            'wl_salmon': mce_best.get('wl_salmon', 0),
            'wl_hawk': mce_best.get('wl_hawk', 0),
            'wl_fox': mce_best.get('wl_fox', 0),
        })

    print("=" * 70)
    print("  K-HIT RATE (MCE-best found in NNUE top-K)")
    print("=" * 70)
    denom = max(len(miss_details), 1)
    for k in K_vals:
        print(f"  K={k:>3}: {pct(hit[k]/denom)} hit   ({hit[k]}/{denom})")
    print()

    # ---------- where does MCE-best actually sit? ----------
    print("=" * 70)
    print("  Distribution of MCE-best NNUE rank")
    print("=" * 70)
    cumulative = 0
    for r in sorted(rank_distribution.keys())[:25]:
        cumulative += rank_distribution[r]
        print(f"  rank {r:>3}: {rank_distribution[r]:>4} positions  (cum: {cumulative}/{denom} = {pct(cumulative/denom)})")
    later = sum(v for r,v in rank_distribution.items() if r >= 25)
    print(f"  rank 25+ : {later:>4} positions")
    print()

    # ---------- per-wildlife when missed ----------
    print("=" * 70)
    print("  Which wildlife does MCE prefer but NNUE ranks out of top-8?")
    print("=" * 70)
    missed = [m for m in miss_details if m['mce_rank_in_nnue'] >= 8]
    hit8 = [m for m in miss_details if m['mce_rank_in_nnue'] < 8]
    missed_wl = Counter(m['mce_best_wildlife'] for m in missed)
    hit_wl = Counter(m['mce_best_wildlife'] for m in hit8)
    all_wl = missed_wl + hit_wl
    print(f"  Wildlife type distribution of MCE-best move:")
    print(f"    {'wildlife':>8} {'missed':>8} {'hit':>8} {'total':>8} {'miss%':>8}")
    for wl in ['bear','elk','salmon','hawk','fox']:
        m = missed_wl.get(wl,0); h = hit_wl.get(wl,0); t = m+h
        miss_pct = 100*m/t if t else 0
        print(f"    {wl:>8} {m:>8} {h:>8} {t:>8} {miss_pct:>7.1f}%")
    print()

    # ---------- What does NNUE top-1 look like when it's wrong? ----------
    print("=" * 70)
    print("  When MCE-best is missed from NNUE top-8, what is NNUE's top-1?")
    print("=" * 70)
    nnue_top1_when_missed = Counter(m['nnue_top1_wildlife'] for m in missed)
    for wl in ['bear','elk','salmon','hawk','fox']:
        print(f"    NNUE picked {wl:>8}: {nnue_top1_when_missed.get(wl,0)} times")
    print()

    # ---------- per-turn-bucket ----------
    print("=" * 70)
    print("  Per-turn K=8 hit rate")
    print("=" * 70)
    print(f"  {'turns':>7} {'n':>6} {'K=1':>7} {'K=4':>7} {'K=8':>7} {'K=16':>7} {'K=24':>7}")
    for lo, hi in [(1,5),(6,10),(11,15),(16,20)]:
        bucket = [m for m in miss_details if lo <= m['turn'] <= hi] if False else None
        bucket = [m for m in miss_details if lo <= pos_turn(m, pos) <= hi]
        if not bucket: continue
        n = len(bucket)
        k_hits = {k: sum(1 for m in bucket if m['mce_rank_in_nnue'] < k) for k in [1,4,8,16,24]}
        print(f"  {lo:2}-{hi:2}   {n:>6} "
              f"{pct(k_hits[1]/n):>7} {pct(k_hits[4]/n):>7} {pct(k_hits[8]/n):>7} "
              f"{pct(k_hits[16]/n):>7} {pct(k_hits[24]/n):>7}")
    print()

    # ---------- gap distribution when missed ----------
    print("=" * 70)
    print("  MCE score gap when NNUE misses (higher = more painful)")
    print("=" * 70)
    if missed:
        gaps = sorted([m['gap'] for m in missed])
        print(f"  n={len(missed)}  min={gaps[0]:.2f}  p25={gaps[len(gaps)//4]:.2f}  "
              f"p50={gaps[len(gaps)//2]:.2f}  p75={gaps[3*len(gaps)//4]:.2f}  max={gaps[-1]:.2f}  "
              f"mean={mean(gaps):.2f}")
        print()

    # ---------- pattern context ----------
    print("=" * 70)
    print("  Board context when NNUE misses (mean wildlife counts)")
    print("=" * 70)
    for label, bucket in [('missed (rank>=8)', missed), ('hit  (rank<8 )', hit8)]:
        if not bucket: continue
        avg = {w: mean([m[f'wl_{w}'] for m in bucket]) for w in ['bear','elk','salmon','hawk','fox']}
        hab = mean([m['hab'] for m in bucket])
        tok = mean([m['tokens'] for m in bucket])
        tl  = mean([m['turns_left'] for m in bucket])
        print(f"  {label}: n={len(bucket)}")
        print(f"    bear={avg['bear']:.1f} elk={avg['elk']:.1f} salmon={avg['salmon']:.1f} "
              f"hawk={avg['hawk']:.1f} fox={avg['fox']:.1f}  hab={hab:.1f} tok={tok:.1f} turns_left={tl:.0f}")
    print()

    # ---------- score breakdown summary ----------
    p0 = [s for s in scores if s['player'] == 0]
    if p0:
        print("=" * 70)
        print(f"  Game score summary ({len(p0)} games, player 0 = MCE-player)")
        print("=" * 70)
        print(f"    base mean={mean([s['base'] for s in p0]):.1f}  "
              f"bonus mean={mean([s['bonus'] for s in p0]):.1f}")


def pos_turn(m, pos):
    return m.get('turn', 0) or next((t for (g,t) in pos if g == m['game']), 0)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'overnight/rank_corr_raw_50g_100r.jsonl'
    # turn is already in each line, so simplify
    # (pos_turn helper is a leftover — the raw line has 'turn')
    main(path)
