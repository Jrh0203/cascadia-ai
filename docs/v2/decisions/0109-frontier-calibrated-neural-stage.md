# ADR 0109: Frontier Calibrated Neural Stage

Status: completed as `calibrated_optimizer_pipeline_invalid`; full trainer,
representation treatment, validation treatment, sealed test, and gameplay
closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-calibrated-neural-stage-v1`

## Context

ADR 0108 repaired the numerical stop rule for ADR 0107 and recombined the
24-group free-residual stage at 96.59% recall and 79.17% exact sets with every
pipeline gate passing. This mechanically authorizes the bounded ADR 0107
neural local-fit stage with the same calibrated monotone AdamW mechanism.

This experiment tests whether that mechanism closes the local fit gap in the
selected full neural model. It is not a full trainer.

## Frozen Evidence

- ADR 0108 combined report BLAKE3:
  `84d59e71f117250546f21118688ec93d40060e39547d464936c7fd4223b8630a`.
- ADR 0108 source bundle BLAKE3:
  `8a33ddf444a3caf7abd7a7450925a9e6b7f5985da12c2e010c3609dd8ddf98ce`.
- The first four ADR 0103 neural local-fit groups.
- The selected model, scale-16 expected-rank objective, target probabilities,
  student temperature 2, residual range, rotation sequence, exposure order,
  network, and all observable inputs.
- Monotone AdamW maximum learning rate
  `2*atanh(0.999)/1200`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, no bias
  correction, weight decay `1e-4`, rate regrowth `2`, 16 half-rate trials,
  minimum accepted rate `1e-8`, and loss tolerance `1e-12`.

No model, objective, group, optimizer constant, target, representation,
dataset, rotation, budget, or gate may change.

## Execution

Continue the selected full model independently on groups 0, 1, 2, and 3 for
at most 1,200 exposures. Cycle rotations `0,1,2,3,4,5` exactly. Evaluate the
unrotated group at exposures 0, 120, 300, 600, and 1,200.

The ADR 0108 numerical stop rule remains part of valid pipeline completion.
If no update is accepted, numerical convergence is valid only when:

- all 16 proposals have finite loss and parameters;
- current parameters, moments, direction, and loss are finite;
- the smallest attempted rate is below `1e-7`;
- no proposal improves current loss by more than `1e-12`; and
- at least one update was previously accepted.

Keep the last accepted model. Do not fabricate exposures.

Run four independent origins through a one-MLX-process-per-host dynamic queue,
then replay every group on a different host. Source identity must match across
john1-john4.

## Gates

The neural pipeline passes only when:

- every group completes 1,200 accepted updates or meets the numerical
  convergence rule;
- every score, loss, parameter, moment, direction, and rate is finite;
- all four origin/replay scientific payloads are bit-identical;
- peak RSS is below 4 GiB with zero process swaps and no attributable positive
  system-swap growth;
- source identity matches on john1-john4; and
- sealed test, gameplay, new teacher compute, cloud, and external compute
  remain closed.

The strength gates are at least 90% aggregate target recall and 75% exact
target sets at both 120 and terminal exposures.

## Mechanical Classification

1. `calibrated_optimizer_pipeline_invalid`
   - any source, replay, resource, finite, completion, or sealed gate fails.
2. `public_observable_representation_insufficient`
   - pipeline passes but terminal strength misses either gate.
3. `full_model_local_budget_insufficient`
   - terminal strength passes but the 120-exposure strength gate fails.
4. `local_failure_not_reproduced`
   - pipeline and both strength checkpoints pass.

Only `local_failure_not_reproduced` authorizes one bounded full-trainer pilot
with this exact optimizer. `public_observable_representation_insufficient`
authorizes one public-observable representation treatment instead. No other
classification authorizes further neural compute.

## Maximum Compute

Exactly four origins, four cross-host replays, source checks, focused/full
tests, one combine, and one report. No alternate optimizer, stop threshold,
learning rate, model, objective, group, budget, representation, full trainer,
validation treatment, sealed test, gameplay, cloud, Modal, or external
compute.

## Result

All four origins and four cross-host replays completed. Every replay scientific
payload was bit-identical, source identity matched across john1-john4, peak RSS
remained below 901 MiB, process swaps were zero, and no task recorded
attributable positive system-swap growth.

The local fits stopped very early:

| Group | Accepted updates | Completion | Recall | Exact sets |
|---:|---:|---|---:|---:|
| 0 | 49 | numerical convergence | 24.32% | 0.00% |
| 1 | 6 | numerical convergence | 43.75% | 0.00% |
| 2 | 8 | completion-rule failure | 24.39% | 0.00% |
| 3 | 1 | numerical convergence | 40.62% | 0.00% |

Group 2 reproducibly failed the frozen completion rule. Its scores, moments,
and accepted rates remained finite and it recorded zero nonfinite rejections,
but it did not satisfy every numerical-convergence condition. No group reached
120 exposures, so that checkpoint is unobserved rather than fabricated.

The terminal descriptive aggregate was 32.39% recall and 0% exact sets. The
pipeline gate fails before a strength or representation classification is
eligible. The mechanical classification is
`calibrated_optimizer_pipeline_invalid`.

The dynamic campaign completed in 31.31 seconds, scheduled 51.58 MLX
process-seconds, averaged 1.65 active processes, peaked at four, and recorded
zero idle process-slot seconds while compatible work was queued. The lower
mean occupancy reflects the preregistered cross-host replay dependency and
group 0's much longer critical path, not queued-work starvation.

The bounded full-trainer pilot and representation treatment are not
authorized. No additional neural compute may proceed under ADR 0109. Any
successor must be separately preregistered from the retained finite failure
evidence.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-calibrated-neural-stage-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-calibrated-neural-stage-v1-result.md`.
