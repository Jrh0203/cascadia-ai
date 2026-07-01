# Complete-Action Frontier Projected-Control Repair V1 Preregistration

ADR 0104 repairs only ADR 0103's insufficient numerical-control iteration
ceiling.

## Frozen Repair

- Same 24 groups, selected scores, float64 objective, exact gradient, box,
  accelerated projected method, and gates.
- Maximum iterations increase from 10,000 to 100,000.
- Four disjoint six-group origin shards run concurrently on john1-john4.
- Four complete shard replays run concurrently in a one-hop ring.
- Up to six worker processes per host are allowed to use the CPU cores
  efficiently.

## Decision

If every convergence, precision, selection, resource, source, and replay gate
passes, the frozen ADR 0103 evidence mechanically selects
`frozen_optimizer_hyperparameters_insufficient`. Otherwise the repair remains
invalid and no treatment is authorized.

## Prohibitions

Do not rerun or alter the analytic, free-AdamW, or neural arms. No extra group,
seed, solver treatment, model treatment, full trainer, validation selection,
sealed test, gameplay, cloud, Modal, or external compute.

Full shard assignment, solver settings, thresholds, replay ring, and resource
gates are authoritative in ADR 0104.
