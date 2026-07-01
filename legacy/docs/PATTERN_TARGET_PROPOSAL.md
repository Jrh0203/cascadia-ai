# Pattern-Target Candidate Augmentation

## Origin

User observation (2026-05-20): MCE rollouts incrementally build up to states,
but in Cascadia "the future is not locally optimal to get there." A
locally-weak move (e.g., placing a single elk that scores ~2 pts) can enable a
high-value future configuration (5-elk line worth 13 pts), and sequential
search loses these moves at the candidate-prefilter stage.

Companion insight: separating tile placement from wildlife placement reveals
upper-bound opportunities. The wildlife layout that would maximize score (if
tiles weren't a constraint) gives a target to approximate with real moves.

## Hypothesis

Augmenting MCE's candidate pool with **pattern-target moves** — moves chosen
specifically because they progress toward high-value future configurations,
regardless of immediate score — exposes patterns that current candidate
generation drops. MCE rollouts then verify which targets are feasible.

## Design

### Module: `crates/cascadia-ai/src/pattern_target.rs`

`pattern_target_candidates(game) -> Vec<ScoredMove>`:

For each wildlife type W ∈ {Bear, Elk, Salmon, Hawk, Fox}:
1. Compute `pattern_target_for_wildlife(board, W, variant, turns_left)`:
   estimate maximum extra W-score achievable over remaining player turns,
   using card-A heuristics:
   - Bear: `(max_extra/2) * 5` (pairs at 5 pts each)
   - Elk: `max_extra * 3` (line extension marginals)
   - Salmon: `max_extra * 4` (run extension marginals)
   - Hawk: `max_extra * 4` (isolated hawk marginal)
   - Fox: `max_extra * 3` (diversity marginal)
2. Find the first move that progresses toward W by:
   - Scanning market pairs for slots that drop wildlife W
   - For each (frontier cell × rotation × wildlife placement), compute
     `delta + pattern_leverage_bonus` where the bonus counts adjacent
     empty wildlife-allowed cells (future pattern slots)
   - Pick the (move, placement) with maximum total
3. Emit a `ScoredMove` with `eval = target_score * EVAL_SCALE` (high enough
   that the prefilter doesn't drop it before MCE rollouts)

### Integration: `mce.rs::expanded_candidates`

```rust
if std::env::var("CASCADIA_PATTERN_TARGET").ok().map(|s| s != "0").unwrap_or(false) {
    candidates.extend(crate::pattern_target::pattern_target_candidates(game));
}
```

Default off. ~5 additional candidates per decision when enabled.

## Why this should work (theory)

- **Bypasses prefilter pruning**: the high synthetic eval ensures these
  candidates survive `nnue_prefilter_candidates` and reach the MCE rollout
  budget.
- **MCE filters wishful patterns**: if the pattern target is infeasible (e.g.,
  bag won't produce enough elk), MCE rollouts will return low values and a
  different candidate wins. The augmentation is essentially "give MCE more
  hypotheses to test."
- **Captures decoupled planning**: the wildlife marginal scoring is computed
  ignoring some tile-placement constraints (use the best wildlife position
  for the chosen tile, not the best tile-and-wildlife pair globally), which
  is the "decouple tile/wildlife" reframe the user proposed.

## Why this might NOT work (risks)

1. **Target estimates may be too optimistic** — the heuristic-based upper
   bounds assume best-case placements that may not all be achievable. MCE
   rollouts should filter, but if many candidates have inflated eval the
   prefilter might select the wrong subset.
2. **Pattern candidates may duplicate existing ones** — `wildlife_strategic_candidates`
   already emits pattern-extending moves. New candidates might overlap.
3. **Wall overhead** — pattern enumeration adds per-decision compute. At
   small MCE budgets the overhead might dominate.

## Bench protocol

Same harness as MCE perf bench (`mce_perf_bench`), with `BENCH_SCORE_MODE=base`
to avoid the bonus-inflation artifact discovered in MCE Perf 3.

Two-row comparison:
- CTRL: `CASCADIA_MCE_TRUNC=3` (champion as of 2026-05-20)
- EXP:  `CASCADIA_MCE_TRUNC=3 CASCADIA_PATTERN_TARGET=1`

100 games × 4 seats = 400 seat-games each side. SE ~0.4 on the difference.
A +0.5 pt or larger mean improvement would be ~1.5σ — directionally compelling
even if not at p<0.05.

## Preliminary results (30g smoke)

| Setting | Wall | Mean (base) | Δ vs CTRL |
|---|---:|---:|---:|
| CTRL TRUNC=3 | 114.22 s | 91.925 | — |
| EXP TRUNC=3 + PATTERN_TARGET | 115.85 s | **92.54** | **+0.62 (z≈1.24)** |

Wall overhead: +1.4% (negligible). Encouraging directional signal at 30 games;
100-game confirmation in progress.

## Expected outcomes

- **+0.5-1.0 pt mean improvement at 100 games**: the experiment succeeds; ship
  CASCADIA_PATTERN_TARGET=1 as the new champion default.
- **+0.0-0.4 pt (within noise)**: pattern candidates aren't adding signal that
  MCE doesn't already extract. Document as null result.
- **Negative**: pattern-target eval is misleading MCE into bad candidates.
  Document and revert.

## Future extensions (if positive)

1. **Tighter pattern targets**: use exact combinatorial enumeration instead of
   linear-extension heuristics. E.g., for elk: enumerate top-K possible lines
   of length 4-5, weight by completion probability.
2. **Multi-wildlife joint targets**: combine across wildlife types (e.g.,
   bear pair + adjacent fox for fox-diversity setup).
3. **Tile-constraint relaxation**: directly implement the user's "ignore
   tile constraints, find ideal wildlife layout, approximate" pipeline as a
   standalone `wildlife_upper_bound_oracle` function used as a candidate
   pre-ranker.
