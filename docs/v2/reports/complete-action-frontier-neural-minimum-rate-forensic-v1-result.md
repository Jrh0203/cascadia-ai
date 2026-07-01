# Complete-Action Frontier Neural Minimum-Rate Forensic V1 Result

Classification: `public_observable_representation_insufficient`.

ADR 0110 used only frozen ADR 0109 JSON and source-bundle evidence. It did not execute MLX, a model forward pass, gradients, or training.

## Proof

- Consistent accepted-rate histories: 6.
- Failed-step starting rate in every history: `9.89635720638073e-05`.
- Smallest of 16 failed-step attempted rates in every history: `3.02012854198631e-09`.
- The smallest attempted rate is below both the `1e-7` convergence threshold and the optimizer's `1e-8` acceptance floor.
- The frozen report records zero nonfinite rejections, finite moments and scores, eight prior accepted updates, and the generic exhausted backtracking failure.
- Frozen source proves every eligible finite loss-nonincreasing proposal would have been accepted.

Therefore the only remaining failed convergence condition was a candidate improvement greater than `1e-12` at a rate below `1e-8`. That proposal was outside the optimizer's eligible update domain and cannot consistently invalidate numerical convergence.

Group 2 is domain-consistently reclassified as numerically converged after eight accepted updates, without changing its model or metrics.

## Recombined Decision

- Terminal target recall: 32.39%.
- Terminal exact sets: 0.00%.
- The 120-exposure checkpoint remains unobserved.

The corrected pipeline passes, but terminal strength misses both gates. The frozen mechanism therefore classifies `public_observable_representation_insufficient` and authorizes one separately preregistered public-observable representation treatment. It does not authorize a full trainer directly.

## Gates

| Gate | Result |
|---|---|
| `domain_consistent_pipeline_passed` | pass |
| `finite_state_proof_passed` | pass |
| `frozen_identity_passed` | pass |
| `frozen_source_logic_passed` | pass |
| `minimum_rate_completion_conflict_proved` | pass |
| `rate_path_enumeration_passed` | pass |
| `strength_checkpoint_observed` | fail |
| `terminal_strength_gate_passed` | fail |
