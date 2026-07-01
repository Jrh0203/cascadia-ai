# Complete-Action Frontier Exact-Float Decimal Control V1 Preregistration

ADR 0106 repairs only ADR 0105's rank input conversion.

## Frozen Correction

- Same first 24 groups and frozen ADR 0103 analytic summaries.
- Same 96-digit breakpoint active-set solver and Decimal gates.
- Convert every fractional expected-rank float exactly with
  `Decimal.from_float`.
- Do not call the float64 analytic or projected control path.
- Run 24 one-group origins and one cross-host replay per group.

## Throughput

Reuse the validated dynamic queue. Origins receive priority and completed
origins immediately unlock replay backfill on a different host. Per-host
capacity ramps only after clean resource telemetry.

## Decision

If every unchanged numerical, selector, identity, resource, and replay gate
passes, the frozen ADR 0103 evidence mechanically selects
`frozen_optimizer_hyperparameters_insufficient`. Otherwise no treatment is
authorized.

## Prohibitions

Do not reinterpret ADR 0105, alter another input, relax a threshold, rerun
frozen AdamW or neural evidence, increase projected iterations, or open a
model treatment, full trainer, validation selection, sealed test, gameplay,
cloud, Modal, or external compute.

ADR 0106 is authoritative for conversion, gates, queue behavior, and maximum
compute.
