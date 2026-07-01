# ADR 0174: Relational Hierarchical Pointer Foundation

Status: preregistered; implementation calibration passed; cross-host production
evidence pending

Date: 2026-06-16

Experiment: `p1-relational-hierarchical-pointer-foundation-v1`

Protocol: `exact-r2-selected-prefix-pointer-alignment-v1`

Research-plan item: P1

## Context

ADR 0114 proved that the exact conditional hierarchy
`draft 16 -> tile 32 -> wildlife 8` can retain 99.18% of the validation
expected-rank target and every validation R4800 winner at 482.4 mean proposals.
ADR 0115 then showed that the first learned implementation was the wrong
representation:

- draft factor recall was 92.84%;
- conditional tile recall was 66.57%;
- integrated proposal recall was 72.48%; and
- the tile top-32 membership gradient opposed the regression/listwise gradient
  at mean cosine `-0.738910`.

Subsequent objective, schedule, capacity, exposure, and local-geometry
treatments did not repair the tile stage. That pointwise flattened-factor
family is closed.

The accepted R2 substrate changes the object being learned. It represents each
board with exact occupied, frontier, habitat-component, and wildlife-motif
tokens. Across the 60,000-row R2 foundation corpus, the maximum was 92 exact
sparse tokens per board, with no clipping or hidden-state leakage.

This is the rigorous interpretation of the proposed 121-cell compression:
P1 will not use 441 dense cells, and its exact sparse active-board support must
fit beneath 121 objects. This does **not** claim that an arbitrary 11-by-11
dense crop is lossless. The sparse representation is exact because it retains
all occupied and legal-frontier coordinates plus the required relational
objects.

## Hypothesis

Every non-anchor complete action in the frozen open corpus can be represented
bijectively as:

```text
structured prelude/draft object
-> exact active-board frontier token + legal tile rotation
-> exact occupied-tile token, selected-prefix new-tile token, or no-placement
```

The selected draft prefix identifies the staged market and public supply. The
selected tile prefix introduces one exact new-tile token before wildlife
destination scoring. The model never needs one flattened row per tile factor
or a 441-cell board tensor.

If this exact mechanical substrate passes, one matched MLX pointer pilot is
authorized. Learned quality remains a separate question.

## Frozen Inputs

- Hierarchical factor cache:
  `artifacts/experiments/full-legal-hierarchical-factor-retrieval-pilot-v1/cache`
- Exact R3/R2 cache:
  `artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache/0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156`
- Exact D6 contract:
  `python/cascadia_mlx/d6_contract_metadata.v1.json`
- Open train: 560 groups and 2,135,111 complete actions.
- Open validation: 240 groups and 860,203 complete actions.
- Champion-frontier anchors remain outside learned hierarchical retrieval,
  exactly as in ADRs 0114 and 0115.

No target, action, split, expected rank, R4800 estimate, frontier anchor, or
hidden-information boundary changes.

## Exact Audit

For each split:

1. Verify every factor-cache shard and every required R3 tensor checksum.
2. Match all group IDs and source action counts.
3. Match every R3 retained action hash to its source action hash.
4. Recompute every draft, draft-plus-tile, and complete-prefix factor hash.
5. Require every tile factor coordinate to identify exactly one active-board
   R2 frontier token.
6. Require every wildlife factor to identify exactly one current occupied
   token, the selected-prefix new tile, or the explicit no-placement sentinel.
7. Require factor maps to be present exactly for non-anchor actions.
8. Reconstruct one pointer identity per non-anchor action and require a
   bijection.
9. Apply all 12 exact D6 transforms and inverses to every spatial pointer and
   tile orientation.
10. Measure the actual token and pointer support distributions.

The audit reads existing immutable caches and writes only a compact JSON
report. It does not emit another action-scale cache.

## Gates

All gates are mandatory:

- 800 open groups and 2,995,314 source actions accounted for exactly;
- every retained cross-cache action hash matches;
- every selected-prefix factor hash matches;
- zero missing or ambiguous tile pointers;
- zero missing or ambiguous wildlife pointers;
- zero action-map failures;
- zero complete-action pointer collisions;
- zero D6 round-trip failures;
- maximum exact active-board sparse tokens at most 121;
- maximum draft pointer support at most 20;
- maximum active-board frontier support at most 31; and
- maximum wildlife destination support at most 25.

The 121-object gate is a compactness and exactness gate, not a claim that 121
dense cells and 121 sparse relational objects carry identical semantics.

## Cross-Host Production Contract

- john2: complete train origin.
- john4: complete validation origin.
- john4: complete train replay after both origins.
- john2: complete validation replay after both origins.

Each split must produce an identical scientific BLAKE3 on two distinct hosts.
Runtime host, elapsed time, and peak RSS are descriptive and excluded from the
scientific identity.

The four audits run only after each host completes its already-queued
opportunity-query and exact-R2 materialization obligations. This backfills the
shorter john2 and john4 lanes while john1 and john3 remain on their longer
critical paths.

## Mechanical Classification

1. `p1_relational_pointer_foundation_structurally_invalid`
   - reports, split coverage, or envelopes are incomplete.
2. `p1_relational_pointer_foundation_cross_host_inconsistent`
   - a split replay differs or does not use a distinct host.
3. `p1_relational_pointer_foundation_failed`
   - production evidence is valid but any exactness or compactness gate fails.
4. `p1_relational_pointer_foundation_passed`
   - every train, validation, replay, and classification gate passes.

Only the passing result authorizes
`matched-mlx-selected-prefix-pointer-pilot`.

## Implementation Calibration

Before freezing the production bundle, the implementation was exercised once
on the complete open validation split. This was an implementation calibration,
not promotion evidence.

It processed:

- 240 groups;
- 860,203 source actions;
- 348,069 tile pointer items;
- 853,003 wildlife pointer items; and
- 14,414,112 D6 coordinate/orientation round-trip checks.

Calibration found zero misses, ambiguities, hash mismatches, action-map
failures, pointer collisions, or D6 failures. The maximum active-board exact
sparse support was 82 tokens, maximum frontier support was 26, and maximum
wildlife destination support was 24.

Production still requires the frozen source bundle and crossed host replays.

## Claim Boundary

A pass proves that the new P1 model can use exact compact pointers and selected
prefixes. It does not prove that the pointer logits are learnable, that the
proposal reaches 98% recall, that gameplay improves, or that Cascadia reaches
100 points.
