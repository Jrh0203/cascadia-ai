# Complete-Action Frontier Arbitrary-Precision Control V1 Preregistration

ADR 0105 independently reconstructs the frozen scale-16 box optimum after the
iterative projected control remained selector-sensitive.

## Frozen Control

- Same first 24 groups, targets, residual box, anchored width-64 selector, and
  frozen ADR 0103 analytic summaries.
- Standard-library Decimal arithmetic at 96-digit precision.
- Independent breakpoint active-set derivation; no float64 analytic or
  projected solver calls.
- Exact Decimal normalization, KKT, objective, selector, and active-set checks.
- Twenty-four independent one-group origins and one cross-host replay each.

## Throughput

Groups are queued individually across john1-john4. Origins receive priority;
completed origins immediately release a replay task that can backfill a free
slot on any different host. The campaign has no coarse host barrier and may
use up to 10 no-swap group processes per host.

## Decision

If every numerical, selector, identity, resource, and replay gate passes, the
frozen ADR 0103 evidence mechanically selects
`frozen_optimizer_hyperparameters_insufficient`. Otherwise the control remains
invalid and no treatment is authorized.

## Prohibitions

Do not rerun or alter the analytic, free-AdamW, neural, or projected arms. No
extra group, precision or solver treatment, threshold relaxation, model
treatment, full trainer, validation selection, sealed test, gameplay, cloud,
Modal, or external compute.

The derivation, thresholds, scheduler behavior, replay requirement, and
resource gates in ADR 0105 are authoritative.
