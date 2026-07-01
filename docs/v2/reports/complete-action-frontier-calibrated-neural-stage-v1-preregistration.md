# Complete-Action Frontier Calibrated Neural Stage V1 Preregistration

ADR 0109 executes the neural local-fit stage authorized by ADR 0108.

## Treatment

- Groups 0, 1, 2, and 3 run independently.
- At most 1,200 exposures per group.
- Frozen rotation cycle `0,1,2,3,4,5`.
- Unchanged selected model, scale-16 objective, calibrated monotone AdamW, and
  ADR 0108 finite numerical-convergence rule.
- One origin per Mac, then one cross-host replay per group.

## Gates

The pipeline must pass every source, replay, resource, finite, completion, and
sealed-boundary gate. Strength must reach at least 90% recall and 75% exact
sets both at 120 exposures and terminally.

Passing both checkpoints classifies `local_failure_not_reproduced` and
authorizes one bounded full-trainer pilot with this exact optimizer. Missing
terminal strength classifies `public_observable_representation_insufficient`.

## Prohibitions

No optimizer, threshold, rate, objective, model, representation, group,
budget, full trainer, validation treatment, sealed test, gameplay, cloud,
Modal, or external compute change.

ADR 0109 is authoritative for the neural groups, completion rule, gates,
classification, queue, and maximum compute.
