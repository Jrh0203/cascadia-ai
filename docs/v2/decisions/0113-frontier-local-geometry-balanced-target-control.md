# ADR 0113: Frontier Local-Geometry Balanced-Target Control

Status: completed as `shared_adapter_capacity_insufficient`; this adapter
parameterization is closed.

Date: 2026-06-16

Experiment ID:
`complete-action-frontier-local-geometry-balanced-target-control-v1`

## Context

ADR 0112 proved that all 11,087 exact ADR 0111 local input rows are unique,
no selected residual is saturated, and independent bounded corrections recover
100% recall and 100% exact target sets in every group. ADR 0111 therefore
failed either because its shared adapter cannot fit the rows or because the
scale-16 expected-rank objective and gradient path dilute the sparse target
signal.

ADR 0112 authorizes exactly one same-representation mechanistic control.

## Frozen Evidence

- ADR 0112 combined report BLAKE3:
  `0802dcbd57b2273134670c547e069483951303165a00f8d4f5cb5c6ecd4bc12a`.
- ADR 0112 source bundle BLAKE3:
  `5e5c815a692a60c2d727100f8d22787765aac58ad66ce6df976621525c107648`.
- ADR 0111 selected model, adapter architecture and seed, local inputs,
  correction equation, bounds, four groups, rotations, optimizer constants,
  exposure ceiling, checkpoints, selector, metrics, and resource gates.

No representation, architecture, initialization, base model, group, target
set, correction range, residual range, rotation, optimizer constant, budget,
selector, metric, or gate may change.

## Treatment

Replace only the scale-16 expected-rank cross entropy with direct balanced
target-membership binary cross entropy on the adapter's pre-tanh scalar logits.

For each group and rotation:

- positives are eligible nonfrontier candidates in the frozen expected-rank
  target set;
- negatives are all other eligible nonfrontier candidates;
- champion-frontier candidates are excluded because the deployed selector
  retains them unconditionally;
- positive loss is the mean `softplus(-logit)`;
- negative loss is the mean `softplus(logit)`; and
- total loss is `0.5 * positive_mean + 0.5 * negative_mean`.

Both classes must be nonempty. There is no auxiliary, regularizer, margin,
weight sweep, threshold, or curriculum.

The unchanged adapter converts logits to
`12*tanh(logit)`, adds the correction to the frozen selected residual, clips
to `[-12, 12]`, and runs the frozen deployed selector.

## Execution

Fit groups 0-3 independently for at most 1,200 accepted updates with rotations
`0,1,2,3,4,5`. Evaluate the unrotated selector at accepted updates 0, 120,
300, 600, and 1,200 when observed, plus the exact terminal state.

Use the unchanged calibrated monotone AdamW and ADR 0110 eligible-domain
numerical convergence rule. Run four distinct origins across john1-john4 and
cross-host replay every group through the dynamic one-process-per-host queue.

## Gates

The pipeline passes only when:

- every group completes 1,200 accepted updates or valid numerical
  convergence;
- zero initialization exactly matches the frozen selected model;
- both target classes are present and every loss, score, parameter, moment,
  direction, and rate is finite;
- all accepted updates are monotone in the balanced objective;
- all four cross-host scientific payloads are bit-identical;
- source identity matches on john1-john4;
- peak RSS is below 4 GiB with zero process swaps and no attributable positive
  system-swap growth; and
- validation, sealed test, gameplay, new teacher compute, cloud, and external
  compute remain closed.

The terminal strength gate is at least 90% aggregate target recall and 75%
exact target sets. A common 120-update aggregate is descriptive only.

## Mechanical Classification

1. `balanced_target_control_pipeline_invalid`
   - any identity, class, equality, finite, monotone, completion, replay,
     resource, or sealed gate fails.
2. `shared_adapter_capacity_insufficient`
   - the valid control misses either terminal strength gate.
3. `expected_rank_gradient_dilution_confirmed`
   - the valid control meets both terminal strength gates.

Only `expected_rank_gradient_dilution_confirmed` authorizes one bounded
objective-corrected multi-group pilot with the same representation.
`shared_adapter_capacity_insufficient` closes this adapter parameterization.
No outcome directly authorizes a full trainer or second representation.

## Maximum Compute

Exactly four origins, four cross-host replays, source checks, focused/full
tests, one combine, and one report. No objective sweep, class-weight sweep,
representation change, architecture change, optimizer treatment, full
trainer, validation treatment, sealed test, gameplay, cloud, Modal, or
external compute.

## Result

All four origins and cross-host replays completed with bit-identical
scientific payloads. Source identity and every resource, finite, equality,
monotone, completion, and sealed-domain gate passed.

| Group | Candidates | Accepted updates | Recall | Exact sets |
|---:|---:|---:|---:|---:|
| 0 | 2,975 | 6 | 40.54% | 0.00% |
| 1 | 4,368 | 3 | 40.62% | 0.00% |
| 2 | 324 | 34 | 100.00% | 100.00% |
| 3 | 3,420 | 13 | 50.00% | 0.00% |

Aggregate terminal recall was 59.86% with 25.00% exact target sets. Direct
balanced supervision therefore did not meet the 90%/75% gate. The sharp
candidate-count split is consistent with insufficient shared adapter capacity
or parameter sharing rather than expected-rank gradient dilution.

The mechanical classification is `shared_adapter_capacity_insufficient`.
The ADR 0111 local adapter parameterization is closed. No full trainer,
objective-corrected pilot, or second representation is authorized.

The campaign completed in 5.37 seconds, scheduled 15.61 process-seconds,
averaged 2.90 active processes, peaked at four, and recorded zero queued-work
idle.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-local-geometry-balanced-target-control-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-local-geometry-balanced-target-control-v1-result.md`.
