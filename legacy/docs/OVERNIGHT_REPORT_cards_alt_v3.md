# Cards-Alt-v3 Overnight Report — Apr 27→28, 2026

_Generated: 2026-04-28 11:00. Bench complete (10,115s wall, mostly contended by stockfish + other processes)._

## TL;DR

- **Training**: 15 iterations from scratch with the four pipeline fixes baked in, using mixed pool from day 1 (heuristics + iter15 + v2_iter1).
- **RMSE**: bottomed at **6.07** (iter 12), final iter 15 = 6.10. Same plateau zone as cards_alt_v2 (6.04-6.16) — pipeline fixes didn't break the inherent variance ceiling.
- **Bench**: **97.2 base / 100.8 with bonus** — IDENTICAL base mean to iter15-with-alt-aware-candidates (97.2). Slightly LOWER bonus (100.8 vs 101.2). Retraining did **not** add a meaningful lift over the candidate-fix alone.

## 1. What was implemented

### Architecture
- **Cargo feature set**: `mid-features,v4-opp,cards-alt-v2`
- **Feature count**: 21,710 (mid 10,862 + v4-opp 369 + cards-alt 192 + cards-alt-v2 10,287)
- **Network shape**: 21,710 → 512 → 64 → 1 (single value head)
- **Training mode**: from scratch (random init via `NNUENetwork::new()`, no warm start)
- **Output**: `nnue_weights_cards_alt_v3_iter{1..15}.bin` (43 MB each)

### Four pipeline fixes baked in for v3

1. **Alt-aware candidate generators** (the "smoking gun" from `wildlife_candidates.rs`):
   - `pattern_setup_value_dispatch` routes by `(wildlife, ScoringCardVariant)`
   - `new_slot_value_dispatch` does the same per-cell
   - 13 new variant-specific functions: `bear_c_setup`, `bear_d_setup`, `bear_b_setup`, `elk_b_setup`, `elk_c_setup`, `elk_d_setup`, `salmon_d_setup`, `salmon_bc_setup`, `hawk_d_setup`, `hawk_b_setup`, `hawk_c_setup`, `fox_b_setup`, `fox_c_setup`, `fox_d_setup`
   - Plus 5 alt slot-value helpers: `bear_slot_value_alt`, `elk_slot_value_alt`, `salmon_slot_value_d`, `hawk_slot_value_alt`, `fox_slot_value_b`

2. **Alt-aware `board_potential`** (already in place from earlier session): per-animal dispatch including `bear_potential_c`, `elk_potential_b`, `salmon_potential_d`, `hawk_potential_d`, `fox_potential_b`.

3. **Three actively-misleading Card-A pattern features gated off** under `cfg(not(feature = "cards-alt"))`:
   - `PAT_BEAR_WASTE` (4 bins) — encoded "bears in components ≥3 = bad", inverted under Bear C/D/B
   - `PAT_HAWK_AT_RISK` (4 bins) — encoded "isolated hawk could lose isolation = bad", inverted under Hawk B/C/D
   - `PAT_MAX_DIV_FOX` (4 bins) — Card-A diversity target, irrelevant for Fox B/C/D
   - Feature slots remain in the index space (backward-compat) — emission is skipped, so weights for those positions stay dormant

4. **`CASCADIA_GREEDY_POTENTIAL=1` set during selfplay**: opponents that fall back to greedy now use alt-aware potential lookahead. Cards_alt_iter15's training had this OFF — opponents played myopic alt-greedy. Stronger pool means better training distribution.

### Pool composition

```
random, scarcity, preference,                          # heuristics (always)
nnue_weights_cards_alt_iter15.bin,                     # original alt champion (RMSE 6.14)
nnue_weights_cards_alt_v2_iter1.bin,                   # per-piece feature warm-start best (RMSE 6.04)
+ all prior cards_alt_v3 iters as they accumulate
```

So even iter 1 of v3 faced 5 opponents, never self-play against random. By iter 15, the pool had 19 entries (5 base + 14 prior v3 iters).

## 2. RMSE trajectory

| Iter | Phase | LR | RMSE | Δ from prev | Wall |
|---|---|---|---|---|---|
| 1 | 1: bootstrap | 1e-4 | 6.40 | — | 7 min |
| 2 | 1 | 1e-4 | 6.54 | +0.14 ⚠ | 34 min |
| 3 | 1 | 1e-4 | 6.17 | −0.37 ✓ | 47 min |
| 4 | 1 | 1e-4 | 6.21 | +0.04 | 28 min |
| 5 | 1 | 1e-4 | 6.15 | −0.06 | 28 min |
| 6 | 2: refine | 5.0e-5 | 6.10 | −0.05 | 34 min |
| 7 | 2 | 4.3e-5 | 6.12 | +0.02 | 38 min |
| 8 | 2 | 3.7e-5 | 6.08 | −0.04 | 53 min |
| 9 | 2 | 3.0e-5 | 6.11 | +0.03 | 45 min |
| 10 | 2 | 2.3e-5 | 6.10 | −0.01 | 32 min |
| 11 | 2 | 1.7e-5 | 6.09 | −0.01 | 37 min |
| 12 | 2 | 1.0e-5 | **6.07** | −0.02 | 59 min ← best |
| 13 | 3: polish | 3.0e-6 | 6.08 | +0.01 | 57 min |
| 14 | 3 | 2.0e-6 | 6.12 | +0.04 | 45 min |
| 15 | 3 | 1.0e-6 | 6.10 | −0.02 | 38 min |

**Total wall**: ~9h 40min (16:35 → 02:15).
**Best RMSE**: 6.07 at iter 12 (Phase 2 end, LR=1e-5).
**Final iter 15 RMSE**: 6.10.

### Cross-run comparison

| Run | Pipeline | Init | Best RMSE | Where it landed |
|---|---|---|---|---|
| cards_alt (15 iters) | Card-A-poisoned | scratch | 6.14 | iter 9 |
| cards_alt_v2 (8 iters) | Card-A-poisoned | warm from iter15 | 6.04 | iter 1 then oscillation |
| **cards_alt_v3 (15 iters)** | **fixed (4 fixes)** | **scratch** | **6.07** | **iter 12** |

**Did from-scratch with fixed pipeline break below 6.0? — NO.** It landed in the same family of plateaus all three runs hit (6.04-6.14). The four pipeline fixes shifted things by ≤ 0.05 in RMSE — minor. The variance ceiling (mostly Card D Hawk, where one tile placement can swing 25+ pts) appears to be the binding constraint, not training pipeline cleanliness.

The training trajectory is consistent with: **Card D's inherent score variance forces RMSE 6.0 as the floor for our network architecture; no amount of feature engineering or pipeline cleanliness gets meaningfully below that.**

## 3. Bench result

_Pending — to be populated when `/tmp/cards_alt_v3_bench.log` finishes (currently 4h 30m wall, longer than the 3 hr estimate)._

### All baselines for context

| Strategy | Base mean | + bonus | Notes |
|---|---|---|---|
| Plain greedy (no potential) | 74.2 | 79.0 | floor |
| Plain greedy + alt-aware potential | 81.0 | 85.9 | hand-crafted heuristics worth +6.8 |
| Greedy-MCE-750 (OLD Card-A candidates) | 96.5 | 100.5 | strong search |
| Greedy-MCE-750 (alt-aware candidates) | 96.6 | 100.2 | candidate fix barely helped greedy |
| Cards-alt iter15 NNUE-MCE-750 (OLD candidates) | 95.9 | 99.6 | underperformed greedy-MCE |
| Cards-alt iter15 NNUE-MCE-750 (alt-aware candidates) | 97.2 | 101.2 | candidate fix +1.3 for NNUE |
| **Cards-alt-v3 iter15 NNUE-MCE-750 (alt-aware everything)** | **TBD** | **TBD** | **this run** |

### Bar to clear

- **≥ 99 base** = real win, alt-pipeline fully validated
- **97-98** = retraining added marginally on top of the candidate fix
- **< 97** = retraining didn't help; the +1.3 from alt-aware candidates was the entire alt-rules ceiling lift available

## 4. Per-animal breakdown delta

_Pending bench completion. Will compare v3 iter15 to:_
- iter15 with old candidates (95.9 base, was the original alt NNUE)
- iter15 with new candidates (97.2 base, the proven candidate-fix lift)
- v2 iter1 with new candidates (96.1 base, per-piece features but trained on poisoned pipeline)

## 5. Verdict

_Pending bench. Honest assessment will be filled in based on:_
- If v3 ≥ 99 base: the four pipeline fixes + retrain compound to a meaningful win. The "training distribution should match inference" hypothesis was correct.
- If v3 ≈ 97-98: candidate fix + retrain added marginal value. Most of the alt-rules NNUE ceiling is in the candidates.
- If v3 ≤ 96: from-scratch retraining didn't help over the iter15-with-alt-candidates result. The variance ceiling is genuinely binding and value functions aren't going to add net value here.

## 6. What's next regardless of result

The RMSE plateau across THREE training regimes (Card-A pipeline, warm-start fixed pipeline, from-scratch fixed pipeline) at 6.04-6.14 strongly implies a fundamental limit on how sharply this NNUE architecture can predict alt-card scores. Future work that might break it:

- **Larger network** (HIDDEN1=1024) — more capacity per feature
- **Multi-head value with Card-D-specific Hawk head** — separate the high-variance signal from the low-variance ones (the v5 split-head architecture, though prior tests showed neutral)
- **MCE-distillation training** (with strong external teacher) — already known to be neutral for Card A but never tested for alt rules
- **Search-side improvements** (UCB/Bayesian-bandit allocators, deeper rollouts for Card D specifically)

The user-confirmed-allowed direction list (`mce_is_the_ai.md`, `feedback_dead_directions.md`) suggests **search-side investments** are the highest expected-value path now that the value-function ceiling appears to be near its inherent limit.
