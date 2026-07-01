# ADR 0157: R5 Component-and-Motif Quotient Foundation

Status: completed; foundation passed

Date: 2026-06-17

Experiment: `r5-component-motif-quotient-foundation-v1`

Protocol: `r5-exact-decoding-and-compactness-v1`

Research-plan item: R5

## Context

R2 established that occupied cells plus legal frontiers are an exact sparse
replacement for a padded dense board. R3 established exact complete-action
edits. R4 showed that a 61-cell focal field with bounded far summaries can be
mechanically exact, but its matched learned comparison selected no bounded
parent representation.

R5 tests a different compression axis. Habitat score and most Card A wildlife
logic are functions of connected components, paths, conflict graphs, and
local opportunities. A model may not need every empty coordinate if those
objects are explicit.

The user's recollection that roughly 121 cells can be sufficient is directionally
consistent with avoiding 441 padded cells, but an exact centered 121-hex disk
does not exist. Centered axial disks contain 61 cells at radius four, 91 at
radius five, and 127 at radius six. R5 therefore avoids an arbitrary 121-slot
crop and tests a semantic quotient directly.

## Decision

Implement one exact Rust census over four information surfaces:

1. `control`: the exact R2 sparse parent and complete R3 action;
2. `quotient`: global, player, market, supply, habitat-component, and
   wildlife-motif objects with no raw empty-coordinate stream;
3. `quotient-local`: the quotient plus the selected tile destination, its six
   directed neighbors, the wildlife destination, and active nature tokens;
4. `hybrid`: the full R2 sparse parent plus all relational objects.

The quotient contains exact score-relevant long-range structure. It is not
allowed to pretend that exact legal affordance follows from aggregate objects.
For every complete action the census records that the quotient alone is
underdetermined, then requires the action-local patch to recover the missing
geometry exactly.

## Relational Objects

Each relative board exposes:

- habitat component membership, size, matching edges, open boundary, cycle
  rank, bridges, articulations, rank, frontier contacts, and merge frontiers;
- Bear components, pairs, singleton completion cells, and oversize risk;
- Elk lines, both extension endpoints, eligible extensions, and overlap;
- Salmon components, endpoints, branch conflicts, validity, and
  continuations;
- Hawk positions, conflict edges, isolated count, and safe opportunities;
- Fox centers, diversity, missing types, and compatible cells; and
- nature tokens.

Global public state, all relative players, market, and exact semantic supply
remain explicit.

## Exact Decoders

The experiment validates:

- current Card A habitat, Bear, Elk, Salmon, Hawk, Fox, nature-token, and base
  score anatomy for all four boards at every position;
- raw-control legal affordance for every complete legal action;
- quotient-only insufficiency for every complete legal action;
- exact legal affordance from quotient plus action-local geometry; and
- exact immediate score delta from the same local surface.

No target, rollout value, learned embedding, or hand-tuned strategic weight
participates.

## Compactness

The report measures distributions for:

- control parent postcard bytes;
- quotient audit postcard bytes;
- local action bytes;
- hybrid audit postcard bytes;
- control model-facing tokens;
- quotient model-facing tokens; and
- hybrid model-facing tokens.

The quotient audit object includes rich variable-length evidence and may be
larger than the compact R2 postcard. The actual serving hypothesis is graph
message count. Material compaction therefore requires either:

```text
quotient median bytes <= 0.80 * control median bytes
or
quotient median model-facing tokens <= 0.80 * control median tokens
```

This rule is frozen before production.

## Production Corpus

| Variable | Value |
|---|---:|
| Host | john1 |
| First seed | `5,110,000` |
| Games | 20 |
| Positions | 1,600 |
| Rayon threads | 6 |
| Rules | four-player AAAAA, no habitat bonus |

Calibration seed `5,100,000` is excluded from production evidence.

## Promotion Rule

Classify `r5_local_geometry_exact_and_quotient_compact` only when:

- all current-score decoders pass;
- raw-control affordance has zero failures;
- quotient-only affordance is underdetermined for every complete action;
- quotient-plus-local affordance has zero failures;
- local immediate score delta has zero failures; and
- the material compactness rule passes.

## Consequences

A pass identifies the minimum exact boundary tested here: long-range
component/motif state plus a small action-local raw patch. It authorizes
matched MLX comparisons among quotient, quotient-local, and hybrid views.

A failure does not revive 441 padded cells. It identifies whether the failed
axis was score semantics, local exactness, or serving compactness.

## Claim Boundary

This ADR can establish exact decoding and compactness. It cannot establish
R4800 retention, learned ranking quality, gameplay strength, or progress to a
100-point mean.

## Outcome

The immutable four-host wave classified R5 as
`r5_local_geometry_exact_and_quotient_compact`.

- 1,600 positions and 2,423,019 complete actions were evaluated;
- 6,400 current-board score decoders had zero failures;
- raw-control, action-local affordance, and local score-delta checks had zero
  failures;
- quotient-only affordance was honestly underdetermined for all 2,423,019
  actions;
- median parent tokens fell from 329 to 196, a 0.595744 ratio;
- P99 parent tokens fell from 495 to 289; and
- the local patch was 44 bytes median and 117 bytes P99.

The verbose quotient audit object was 2.628834x the control postcard at the
median, confirming why audit serialization is not the serving representation.
The model-facing graph surface passed the frozen compactness rule.

R5 quotient plus action-local geometry advances to a matched MLX comparison.
See `docs/v2/reports/r5-component-motif-quotient-foundation-v1-result.md`.
