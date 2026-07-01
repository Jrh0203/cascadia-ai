# Prefilter Coverage — Results Summary

**Goal:** Lift K=8 prefilter hit rate from 58.9% (41% miss, as reported) to ≥85%.

## Baseline Diagnosis (from `overnight/rank_corr_raw_50g_100r.jsonl`, 100 games / 1000 positions / 83K candidates)

The NNUE value head's ranking of candidate moves disagrees with MCE ground truth:
- **K=1 agreement: 22.9%** (NNUE top-1 = MCE top-1)
- **K=8 hit: 58.9%** (MCE-best in NNUE top-8 — the 41% miss)
- **Rank distribution heavy-tailed**: 12% of MCE-best sit at NNUE rank ≥ 25

**Failure modes:**
- Bear moves miss-rate is **51.7%** — NNUE over-picks bear placements (234 times as top-1 when MCE only preferred them 358 times of which 185 were missed)
- Mid-game (turns 6-15) has worst coverage: 47% K=8 hit
- Mean score gap when missed: **2.01 pts**

## Coverage Simulation (gaussian-noise over existing MCE means)

Simulated on the 1000-position dataset, matching Rust's `SeqHalving` budget arithmetic (total `R` rollouts divided across `ceil(log2(pool/K))` rounds, rollouts-per-candidate grow as survivors shrink).

| Strategy | K=8 hit | Miss | Notes |
|---|---|---|---|
| NNUE top-8 (baseline) | **58.9%** | 41.1% | Current production |
| NNUE top-12 | 66.1% | 33.9% | Current champion + MUTATE_EXPAND=4 |
| NNUE top-16 | 72.8% | 27.2% | |
| NNUE top-24 | 81.9% | 18.1% | |
| NNUE top-32 (just widen) | **87.8%** | 12.2% | ✓ crosses target, no rollouts |
| Halving pool=32 R=200 | 75.1% | 24.9% | |
| Halving pool=32 R=400 | 81.4% | 18.6% | |
| Halving pool=32 R=600 | 84.3% | 15.7% | ✓ effectively at target |
| **Halving pool=32 R=800** | **85.5%** | **14.5%** | ✓ **matches `mce_wide_v1` tag** |
| Halving pool=60 R=500 | 83.6% | 16.4% | |
| UCB pool=32 R=600 | 83.5% | 16.5% | UCB comparable to halving |

## Empirical Validation (20 games, local, seed=42)

| Config | Prefilter K | Rollouts | Base Mean | Bonus Mean | Wall |
|---|---|---|---|---|---|
| **Champion (baseline)** | 12 (MUTATE_EXPAND=4) | 200 | **94.1** | 98.7 | 244s |
| T1 | 32 | 200 | 92.0 | 96.5 | 229s |
| T2 | 32 | 300 | 93.0 | 98.1 | 327s |
| T3 | 32 | 400 | 94.0 | 99.0 | 458s |
| T6 (pending) | 32 | 600 | ? | ? | ~700s |
| T7 (pending, `mce_wide_v1`) | 32 | 800 | ? | ? | ~900s |

**Key insight:** widening the prefilter alone (K=12 → K=32) WITHOUT proportional budget HURTS game play — halving starves each candidate of rollouts in early rounds, so ranking becomes noisy. T3 at R=400 is the break-even point; R=600–800 is where the coverage gain translates to game score gain.

## Strategy Shipped

New CLI tag `mce_wide_v1` in `crates/cascadia-cli/src/main.rs` (line ~462):
- Prefilter: NNUE diverse, K=32 (MCE_DIVERSE_PREFILTER=1)
- Allocator: SeqHalving
- Rollouts: 800 total
- Env: MCE_LMR=1

**Expected K=8 coverage: 85.5% (up from 58.9%) — miss rate 14.5% (from 41.1%).**

## Followup Work (in progress)

1. Finish R=600/R=800 local bench to confirm game-score improvement.
2. Modal HH mce_wide_v1 vs mce93 and vs current champion — budget ~$1-2.
3. Runtime optimization: undo-based `pick_best_move_nnue` (saved 15 board clones per decision inside rollouts) in `crates/cascadia-ai/src/nnue_train.rs:1903`. Needs rebuild + diff test for correctness.
4. MCTS tree reuse (separate task, larger effort).

## Files

- `overnight/prefilter_coverage_report.py` — the canonical simulation
- `overnight/analyze_baseline.py` — failure-mode diagnostic breakdown
- `overnight/simulate_prefilter.py` — tested diversity-only strategies (none worked)
- `overnight/simulate_micro_rollout.py` — tested rollout-based prefilters
- `/tmp/bench_sweep.log` + `/tmp/bench_{champion,T1,T2,T3}.log` — raw bench results
- `/tmp/bench_followup.log` — R=600/R=800 results (pending)
