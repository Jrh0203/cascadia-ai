# R5 Component-and-Motif Quotient Foundation V1 Preregistration

Date: 2026-06-17

ADR: 0157

Experiment: `r5-component-motif-quotient-foundation-v1`

Protocol: `r5-exact-decoding-and-compactness-v1`

Status: frozen before production

## Question

Can exact habitat-component and wildlife-motif objects preserve Card A score
semantics while a small action-local raw patch recovers every tested legal
affordance and immediate score delta, at materially lower model-facing parent
cost than exact R2?

## Frozen Corpus

```text
host: john1
first seed: 5,110,000
games: 20
positions: 1,600
rayon threads: 6
rules: four-player AAAAA, no habitat bonus
```

Seed `5,100,000` was used only for implementation calibration and is excluded.

## Frozen Comparisons

- exact R2 sparse parent control;
- component-and-motif quotient;
- quotient plus action-local geometry;
- exact R2 plus relational graph hybrid.

The action-local patch contains only the tile destination, six directed
neighbors, optional wildlife site, and active nature tokens.

## Frozen Gates

```text
current score decoder failures == 0
control affordance failures == 0
quotient underdetermined count == complete actions
local affordance failures == 0
local score-delta failures == 0
quotient/control median tokens <= 0.80
  OR quotient/control median canonical message bytes <= 0.80
```

Every complete legal action participates. No action subsampling is allowed.

## Predictions

1. Quotient-only state will preserve current score but not exact local
   affordance.
2. The local patch will recover exact affordance and score delta.
3. Quotient model-facing tokens will be below 80% of exact R2.
4. The verbose audit serialization may be larger than R2 despite a smaller
   graph-serving token surface.

## Invalidators

- source bundle or executable mismatch;
- any calibration seed in production;
- fewer than 1,600 positions;
- missing complete actions;
- scientific hash mismatch;
- altered 0.80 threshold after launch; or
- hidden target- or rollout-derived features.

## Claim Boundary

A pass authorizes matched learned R5 ablations. It does not select a final
representation or claim gameplay improvement.
