# Complete-Action Frontier Local-Geometry Balanced-Target Control V1 Preregistration

Date: 2026-06-16

Experiment ID:
`complete-action-frontier-local-geometry-balanced-target-control-v1`

## Question

Does ADR 0111 fail because expected-rank supervision dilutes sparse target
gradients, or because the unchanged shared local-geometry adapter cannot fit
the four groups?

## Frozen Control

- Same selected base, adapter architecture and initialization, observable
  inputs, correction equation, score bounds, groups, rotations, calibrated
  optimizer, 1,200-update ceiling, selector, and strength gates.
- Only the objective changes.
- Balanced BCE on pre-tanh adapter logits:
  `0.5*mean(softplus(-positive)) + 0.5*mean(softplus(negative))`.
- Positives are frozen target-set candidates; negatives are all other
  eligible nonfrontier candidates; frontier candidates are excluded.
- No auxiliary, sweep, regularizer, margin, or curriculum.

## Cluster Execution

Four distinct origins run across john1-john4, followed by cross-host replay.
The dynamic queue runs one optimizer process per host, resumes missing work
only, and records utilization and queued-work idle.

## Decision Rule

- Invalid pipeline: `balanced_target_control_pipeline_invalid`.
- Valid control below 90% recall or 75% exact sets:
  `shared_adapter_capacity_insufficient`.
- Valid control meeting both gates:
  `expected_rank_gradient_dilution_confirmed`.

No outcome directly authorizes a full trainer or second representation.
