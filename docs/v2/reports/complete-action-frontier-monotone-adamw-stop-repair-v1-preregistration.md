# Complete-Action Frontier Monotone AdamW Stop-Rule Repair V1 Preregistration

ADR 0108 repairs only the five ADR 0107 groups that exhausted finite
backtracking after reaching float32 numerical saturation.

## Repair

- Groups: `0,2,8,14,23`.
- Same optimizer, maximum rate, moments, objective, model, and 1,200-update
  ceiling.
- Exhausted finite backtracking below `1e-7` records numerical convergence
  instead of failure.
- Keep the last accepted parameters; do not invent updates.
- Replay each repaired group on a different host.
- Reuse the other 19 ADR 0107 groups byte-for-byte.

## Decision

If the recombined 24 groups retain at least 95% recall, 75% exact sets, and
pass every pipeline gate, ADR 0107 neural Stage 2 becomes authorized.
Otherwise no neural or model treatment may launch.

## Prohibitions

No rerun of completed groups, alternate stop threshold, optimizer or learning
rate change, representation, neural work before the gate, full trainer,
validation treatment, sealed test, gameplay, cloud, Modal, or external
compute.

ADR 0108 is authoritative for the repair groups, convergence rule, gates,
queue, and maximum compute.
