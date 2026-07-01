# Conditional Tile Local-Geometry Dropout V1 Preregistration

Date: 2026-06-16

Decision: [ADR 0124](../decisions/0124-conditional-tile-local-geometry-dropout.md)

Experiment ID: `conditional-tile-local-geometry-dropout-v1`

Preflight ID: `conditional-tile-local-geometry-dropout-preflight-v1`

## Purpose

Prepare the single targeted structural-regularization successor selected by
ADRs 0121-0123 without changing or delaying the active ADR 0120 origin.

Training remains contingent on a valid
`optimizer_schedule_tile_insufficient` result. A sufficient ADR 0120 result
cancels this branch without training.

## Frozen Difference

Relative to ADR 0120, change only training-time tile local-geometry columns
`[8,188)`. On every epoch and query, select the deterministic hash-ranked half
of items and cyclically rotate those local blocks within the query. The
selection changes deterministically by epoch. All other features, labels,
architecture, objective, optimizer, learning-rate schedule, model seed, batch
order, data, checkpoint selection, validation, and inference remain fixed.

The frozen rate is 50%. No rate or mechanism sweep is allowed.

## Preflight Gates

Before branch authorization:

- exact corruption scope and selected counts;
- 200-epoch per-item coverage in `[0.30,0.70]`;
- cross-host epoch-one selection digest equality;
- finite, nonzero, changed optimization signal;
- complete batch coverage with at most 50% preparation overhead;
- peak process RSS below 4 GiB and zero swaps; and
- train cache only, with validation, test, gameplay, teacher, cloud, and
  external compute closed.

## Strength Gates

If training is authorized:

- train tile recall strictly above 95%;
- validation tile recall strictly above 90%;
- mixed validation target recall strictly above 98%;
- mixed validation R4800 winner retention strictly above 98%; and
- every ADR 0115 integrated proposal gate passes.

Pipeline integrity has precedence over strength. One origin only.
