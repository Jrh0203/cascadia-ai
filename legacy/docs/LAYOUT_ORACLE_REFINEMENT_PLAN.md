# Layout-Oracle Refinement Plan

User insight (2026-05-20): enumerate C(20, 5) = 15,504 hypothetical 5-wildlife
configurations from the current board, score them as joint end-states, then pick
moves toward the best-scoring achievable layout. This bypasses MCE's
incremental greedy nature by directly evaluating multi-placement synergies
(e.g., 2 bears that form a pair score 5 pts together but 0 individually).

## v1 result: null at 100 games

```
CTRL TRUNC=3                     : mean(base) = 92.060
EXP TRUNC=3 + LAYOUT_ORACLE=1    : mean(base) = 92.162
Δ = +0.102 (z ≈ 0.24, within noise)
```

Wall overhead +1.2%. Tests pass. Code at `crates/cascadia-ai/src/layout_oracle.rs`.

## Why v1 was null — diagnosis

Five compounding issues, in order of severity:

### 1. Stale habitat scoring
Synthesized frontier cells didn't update `largest_group`. The hypothetical
board reported habitat as if the frontier placement didn't add any cell.
Top-K layouts were thus selected purely on wildlife pattern, ignoring habitat.

### 2. Universe was speculative
Frontier cells were treated as freely tileable with any terrain via a Forest
placeholder. In reality the tile that goes there is constrained by bag
contents + edge compatibility. Many "universe" placements were unrealizable
fantasies.

### 3. No achievability weighting
A layout requiring 3 specific salmon with only 4 left in bag was scored equal
to a layout requiring 3 elk with 18 left. Top-K extraction was scoring-only,
not feasibility-aware.

### 4. Single-move extraction lost plan coherence
For each top layout, the FIRST achievable placement was emitted as a
candidate. But playing that move could preclude the OTHER 4 placements in the
layout (e.g., tiling a cell that one of them needed empty).

### 5. Top-K converged on the same first move
The highest-leverage single placement appeared in most top-K layouts.
Extracting "first achievable" from each gave 8 candidates pointing to 1
unique move. Diversity = effectively zero.

## v2 refinements (implemented)

### Universe construction
- **A1**: Dropped synthesized frontier cells entirely. Universe now =
  already-placed empty wildlife slots only.
- Removes the fantasy issue and the stale-habitat issue at the cost of
  missing plans that require tiling a new cell first.

### Layout scoring  
- Now exact via real `with_wildlife` + revert on already-placed cells.
- Wildlife score is correct. Habitat is invariant across all hypothetical
  layouts in v2 mode (no tile placement during scoring), so argmax is
  numerically clean.

### Gating
- `CASCADIA_LAYOUT_ORACLE=1` enables the module.
- `CASCADIA_LAYOUT_ORACLE_V2=1` (default on when oracle enabled) selects
  the strict-mode universe.
- Default off; preserves prior behavior bit-exactly when disabled.

## v2 smoke: encouraging

30g preview: Δ = +0.25 (vs +0.058 for v1). Wall slightly faster (smaller
universe → less compute per call). 100g result pending.

## v3 refinements (planned, NOT implemented)

If v2 100g shows Δ ≥ +0.2:

### B1 — Full habitat-aware scoring (Layer 1 from earlier)
Re-introduce frontier cells via REAL place_tile / undo with best-matching
terrain selected from the bag, not a Forest placeholder. Pricier per layout
(~5× slower) but unlocks plans requiring tiling.

### C1 — Achievability weighting
Per-placement: `feasibility = min(bag_W / required_W, 1) × min(turns_left / 5, 1)`.
Layout score multiplied by joint feasibility (product across 5 placements).

### D1 — Plan-aware move extraction  
For each top layout, scan ALL market slots; find the move that:
1. Achieves one placement in the layout
2. Leaves the other 4 placements still reachable
Score by `progress_score = num_achievable_remaining + base_eval`.

### D2 — Diversity enforcement
Top-K candidates must point to ≥K-2 distinct (market_idx, tile_q, tile_r)
tuples. Dedupe upstream and re-extract from next layout when conflicts.

### E1 — Importance sampling (perf)
For universes larger than 15 placements (C(15,5) = 3,003), sample 500
layouts weighted by `pattern_leverage` instead of full enumeration.
Keeps wall overhead bounded at high universe sizes.

### G1 — Per-pattern-class diagnostic
When the oracle wins, break down by wildlife type. Useful for identifying
where the multi-placement-synergy benefit comes from.

## v4 refinements (alternative paradigm)

If v2 + v3 still null, the bottleneck is conceptual not implementation:

### F1 — Training-time fix
Train NNUE to predict score-at-5-turns-from-now rather than
score-at-game-end. With this, NNUE itself recognizes multi-step setups
and MCE rollouts at TRUNC=3 already capture the value — no inference-time
augmentation needed.

This is multi-week training work but architecturally cleaner.

## Realistic outcome bounds (from "MCE Is The AI" empirics)

| Outcome | Probability | Decision |
|---|---:|---|
| Δ ≥ +0.5 (significant lift) | 10% | Ship as champion default |
| +0.2 ≤ Δ < +0.5 (small real signal) | 30% | Ship as optional flag |
| -0.1 ≤ Δ < +0.2 (null) | 45% | v3 refinements; if still null, pivot to v4 (training) |
| Δ < -0.1 (regression) | 15% | Debug then revert |

## Files

- `crates/cascadia-ai/src/layout_oracle.rs` — implementation
- `crates/cascadia-cli/examples/mce_perf_bench.rs` — bench harness with
  `BENCH_SCORE_MODE=base` for bonus-inflation-free comparisons
- `docs/PATTERN_TARGET_PROPOSAL.md` — sister experiment (v0 of pattern-aware
  candidate augmentation; null at 100g)
- This file — comprehensive refinement plan
