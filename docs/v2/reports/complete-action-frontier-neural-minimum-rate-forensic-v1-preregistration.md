# Complete-Action Frontier Neural Minimum-Rate Forensic V1 Preregistration

ADR 0110 audits ADR 0109 group 2 using only frozen optimizer summaries and
source logic.

## Method

- Enumerate every accepted-rate path consistent with eight updates, 13 total
  backtracks, maximum backtrack 4, and the frozen minimum, maximum, and mean
  accepted rates.
- Reconstruct the failed-step starting rate and all 16 attempted rates.
- Use frozen finite-state and failure-path evidence to identify the only
  rejected convergence subcondition.
- Do not run MLX, a model forward pass, gradients, or training.

## Decision

If every consistent path proves that an improving proposal existed only below
the optimizer's frozen `1e-8` acceptance floor, group 2 is domain-consistently
reclassified as numerically converged with its frozen final model and metrics.
The four frozen terminal groups are then mechanically classified.

## Prohibitions

No neural rerun, threshold, optimizer, model, objective, representation,
metric, full trainer, validation treatment, sealed test, gameplay, cloud,
Modal, or external compute.

ADR 0110 is authoritative for the forensic proof, reclassification, gates,
and maximum compute.
