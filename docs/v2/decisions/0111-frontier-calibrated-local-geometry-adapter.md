# ADR 0111: Frontier Calibrated Local-Geometry Adapter

Status: completed as `calibrated_local_geometry_insufficient`; the single
ADR 0110 representation treatment is exhausted and no full trainer is
authorized.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-calibrated-local-geometry-adapter-v1`

## Context

ADR 0110 proved that the corrected ADR 0109 pipeline was valid but reached
only 32.39% target recall and 0% exact target sets. It mechanically classified
the selected public-observable representation as insufficient and authorized
exactly one public-observable representation treatment on the same four
groups.

The strongest previously tested raw observable construction was the exact
rotation-canonical local relation path from ADR 0088/0098. It was evaluated
under the superseded optimizer and full-model training regime, so it has not
been tested as an isolated representation treatment with the calibrated
optimizer.

## Frozen Evidence

- ADR 0110 combined report BLAKE3:
  `ce392cb3b0be15e3b3208b04f7fc186779a1aa0b135ed58851025df195a4b9ae`.
- ADR 0109 selected model, first four local-fit groups, scale-16 expected-rank
  objective, target probabilities, student temperature 2, residual range,
  rotation sequence, exposure order, and all non-treatment observable inputs.
- ADR 0108 calibrated monotone AdamW constants and ADR 0110's
  domain-consistent numerical convergence rule.
- ADR 0088 local relation schema `active-board-local-13-v1`, canonical action
  elision, 192-dimensional local hidden width, and mean/max candidate-set
  pooling topology.

No dataset, target, objective, group, optimizer constant, base model,
non-treatment input, rotation, metric, budget, or gate may change.

## Treatment

Freeze the selected ADR 0081 model and its predictions. Add exactly one
trainable residual adapter whose public-observable input is:

- the 13 exact active-board relations in the candidate tile's
  rotation-canonical frame;
- the canonical action features with absolute coordinates and orientation
  removed;
- the frozen candidate prior features; and
- the frozen public global features.

The adapter topology is the ADR 0088 local path:

- input projection `D -> 384 -> 192` with GELU and LayerNorm;
- masked candidate mean and maximum pooling;
- output projection `768 -> 576 -> 192` with GELU and LayerNorm; and
- a zero-initialized scalar residual head.

The correction is `12 * tanh(head)` and the total residual is
`clip(frozen_base_residual + correction, -12, 12)`. Zero initialization must
make every initial score bit-identical to the frozen selected model.

Only adapter parameters are trainable. The frozen base may not receive
gradients or updates.

## Execution

Fit groups 0, 1, 2, and 3 independently for at most 1,200 accepted updates.
Cycle rotations `0,1,2,3,4,5` exactly. Evaluate each unrotated group at
accepted updates 0, 120, 300, 600, and 1,200 when observed, plus the exact
terminal accepted state.

Use the frozen calibrated monotone AdamW. Numerical convergence is valid when:

- all 16 proposals have finite loss and parameters;
- current parameters, moments, direction, and loss are finite;
- the smallest attempted rate is below `1e-7`;
- no proposal in the eligible update domain, rate at least `1e-8`, improves
  current loss by more than `1e-12`; and
- at least one update was previously accepted.

Below-minimum diagnostic proposals are retained in telemetry but cannot
invalidate convergence. Keep the last accepted model and never fabricate an
unobserved checkpoint.

Run four independent origins through a dynamic one-MLX-process-per-host queue
across john1-john4, then replay every group on a different host. The queue must
resume missing work only, avoid fixed barriers, and record utilization and
queued-work idle time.

## Gates

The pipeline passes only when:

- every group completes 1,200 accepted updates or meets the corrected
  numerical convergence rule;
- the frozen-base equality gate passes at update zero;
- every score, loss, parameter, moment, direction, and rate is finite;
- all four origin/replay scientific payloads are bit-identical;
- source identity matches on john1-john4;
- peak RSS is below 4 GiB with zero process swaps and no attributable positive
  system-swap growth; and
- sealed test, gameplay, new teacher compute, cloud, and external compute
  remain closed.

The terminal strength gate is at least 90% aggregate target recall and 75%
exact target sets. Any observed 120-update aggregate is descriptive only;
early numerical convergence must not fabricate it.

## Mechanical Classification

1. `calibrated_local_geometry_pipeline_invalid`
   - any identity, equality, replay, resource, finite, completion, or sealed
     gate fails.
2. `calibrated_local_geometry_insufficient`
   - the pipeline passes but terminal strength misses either gate.
3. `calibrated_local_geometry_sufficient`
   - the pipeline and terminal strength gate pass.

Only `calibrated_local_geometry_sufficient` authorizes one bounded full-trainer
pilot with this exact frozen-base adapter, objective, and optimizer.
`calibrated_local_geometry_insufficient` exhausts the single representation
treatment authorized by ADR 0110 and authorizes no second representation
treatment or full trainer.

## Maximum Compute

Exactly four origins, four cross-host replays, source checks, focused/full
tests, one combine, and one report. No second representation, architecture
sweep, width sweep, optimizer treatment, objective treatment, full trainer,
validation treatment, sealed test, gameplay, cloud, Modal, or external
compute.

## Result

All four origins and all four cross-host replays completed with bit-identical
scientific payloads. The zero-initialized adapter matched the frozen selected
model exactly in all six rotations for every group. Source identity matched
across john1-john4, peak RSS remained below 891 MiB, process swaps were zero,
and no task recorded attributable positive system-swap growth.

Every group met the domain-consistent numerical convergence rule:

| Group | Accepted updates | Recall | Exact sets |
|---:|---:|---:|---:|
| 0 | 4 | 24.32% | 0.00% |
| 1 | 31 | 78.12% | 0.00% |
| 2 | 307 | 100.00% | 100.00% |
| 3 | 26 | 81.25% | 0.00% |

The aggregate reached 71.13% target recall and 25.00% exact target sets.
No common 120-update checkpoint was observed, so it remains descriptive and
unfabricated. The terminal 90% recall and 75% exact-set gate failed.

The mechanical classification is
`calibrated_local_geometry_insufficient`. Exact rotation-canonical local
geometry is useful but not sufficient as the single isolated public-observable
adapter. ADR 0110's one representation treatment is exhausted. A second
representation treatment and full trainer are not authorized.

The dynamic campaign completed in 10.97 seconds, scheduled 24.26 MLX
process-seconds, averaged 2.21 active processes, peaked at four, and recorded
zero idle slot-seconds while compatible work was queued. The collection
epilogue exposed and permanently fixed a scheduler assumption that every host
would own both an origin and replay directory; completed experiment work was
preserved.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-calibrated-local-geometry-adapter-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-calibrated-local-geometry-adapter-v1-result.md`.
