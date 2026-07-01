# ADR 0107: Frontier Calibrated Monotone AdamW

Status: original Stage 1 completed as
`calibrated_optimizer_pipeline_invalid`; ADR 0108 repaired the stop rule and
authorized neural Stage 2. Sealed test and gameplay remain closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-calibrated-monotone-adamw-v1`

## Context

ADR 0106 independently proved that the frozen scale-16 objective and residual
box recover 100% of all 24 target sets. Frozen free-parameter AdamW at learning
rate `1e-4` reaches only 59.22% recall and zero exact sets after 1,200 updates.
The mechanical classification is
`frozen_optimizer_hyperparameters_insufficient`.

Exactly one calibrated local optimizer mechanism is authorized before any
representation change or full trainer.

## Frozen Inputs

- The same first 24 free-residual groups and first four neural local-fit groups
  from ADR 0103.
- The same selected checkpoint, scale-16 objective, target probabilities,
  student temperature 2, residual range `+/-12`, rotations, exposure order,
  Adam first and second moment coefficients, epsilon, and weight decay
  `1e-4`.
- ADR 0106 combined report BLAKE3:
  `85cb6e36db5372ef2cd6910dd9917068d2d1c61d50415a20446f65c8c08e230b`.
- Cohort digest:
  `30899dec701f053d96023f963b473681516fb0df00a58edf54146c623fd2769d`.

No objective, target, representation, model width, dataset, group, rotation,
weight decay, exposure budget, validation treatment, sealed domain, or
gameplay behavior may change.

## Optimizer Mechanism

Use one custom AdamW mechanism with frozen Adam moments and decoupled weight
decay, plus deterministic monotone backtracking on the current batch.

The maximum learning rate is:

`2 * atanh(0.999) / 1200 = 0.006333668612083666`.

This is the normalized raw-parameter step required to traverse a tanh residual
from `-99.9%` to `+99.9%` of its score box in 1,200 sustained Adam steps. The
frozen `1e-4` rate can traverse only 0.12 raw units in the same budget.

For each update:

1. Compute the ordinary AdamW first moment, second moment, and direction using
   beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, no bias correction, and weight
   decay `1e-4`.
2. Start from the lesser of the maximum rate and twice the previous accepted
   rate.
3. Evaluate the proposed parameters on the same current batch.
4. Accept the first finite proposal whose loss does not exceed the pre-update
   loss by more than `1e-12`.
5. Otherwise halve the rate, for at most 16 trials and no lower than `1e-8`.
6. Failure to accept a step invalidates that group.

The same code and constants must serve free residuals and the full neural
model. Record accepted learning rate, backtracks, rejected nonfinite proposals,
loss monotonicity, and optimizer state finiteness.

## Stage 1: Free-Residual Gate

Run the first 24 groups independently for 1,200 updates. Evaluate at updates
0, 6, 24, 60, 120, 300, 600, and 1,200.

Stage 1 passes only with:

- at least 95% aggregate target recall at 1,200 updates;
- at least 75% exact target sets at 1,200 updates;
- finite scores, losses, parameters, moments, and accepted rates;
- every update accepted within the frozen backtracking budget;
- peak RSS below 4 GiB, zero process swaps, and no attributable positive
  system-swap growth;
- identical source on john1-john4; and
- one bit-identical scientific replay of every group on a different host.

If Stage 1 fails, stop. Neural continuation is not authorized.

## Stage 2: Neural Local-Fit Gate

Only after Stage 1 passes, continue the selected full model independently on
the first four groups for 1,200 exposures each. Cycle rotations
`0,1,2,3,4,5` exactly and evaluate at 0, 120, 300, 600, and 1,200 exposures.

Stage 2 passes at 1,200 exposures with at least 90% aggregate target recall and
75% exact sets, plus the same finite, step-acceptance, resource, source, and
cross-host replay gates.

## Mechanical Classification

1. `calibrated_optimizer_pipeline_invalid`
   - any source, replay, resource, finite-value, step-acceptance, or sealed
     domain gate fails.
2. `calibrated_optimizer_mechanism_insufficient`
   - Stage 1 is valid but misses its recall or exact-set gate.
3. `public_observable_representation_insufficient`
   - Stage 1 passes but Stage 2 fails at 1,200 exposures.
4. `full_model_local_budget_insufficient`
   - Stage 2 fails at 120 exposures and passes at 1,200.
5. `local_failure_not_reproduced`
   - Stage 2 already passes at 120 exposures.
6. `local_optimizer_mechanism_confirmed`
   - Stage 2 improves materially but no classification above applies.

Only a passing neural classification authorizes a bounded full-trainer pilot
with this exact optimizer. A representation classification authorizes one
public-observable representation treatment instead.

## Cluster And Throughput

All optimizer tasks use MLX, so run at most one optimizer process per host.
Use a dynamic one-group queue across john1-john4:

- Stage 1: 24 unique free-residual origins, then cross-host replay backfill.
- Stage 2 if authorized: four unique neural origins, then cross-host replay
  backfill.
- Origins have priority; no host barrier is required.
- Resume only missing atomic tasks.

Record queue snapshots, per-host task counts, critical path, host-seconds,
confirmation fraction, idle time with compatible work queued, duplicate
discovery fraction, memory, swap, and accepted-rate distributions.

## Maximum Compute

Stage 1 is exactly 24 origins plus 24 replays. If and only if it passes, Stage
2 is exactly four origins plus four replays. Add focused/full tests, source
checks, and one report. No optimizer sweep, alternate maximum rate,
backtracking factor, weight decay, budget, group, seed, representation, full
trainer, validation treatment, sealed test, gameplay, cloud, Modal, or
external compute.

## Result

Stage 1 completed all 24 free-residual origins and all 24 cross-host replays.
Every retained scientific payload reproduced bit-identically, source identity
matched across john1-john4, peak process RSS remained below 893 MiB, process
swaps remained zero, and no task recorded attributable positive system-swap
growth.

The optimizer mechanism passed its terminal strength gate:

- at 120 updates: 96.24% recall and 70.83% exact sets;
- terminal: 96.59% recall and 79.17% exact sets; and
- all 851 target slots and 24 groups remained finite.

The pipeline gate failed because five groups could not accept another strictly
loss-nonincreasing update before 1,200 steps:

| Group | Accepted updates | Recall | Exact | Minimum accepted rate |
|---:|---:|---:|---:|---:|
| 0 | 916 | 100.00% | yes | `9.664e-8` |
| 2 | 1,105 | 100.00% | yes | `4.832e-8` |
| 8 | 767 | 97.67% | no | `4.832e-8` |
| 14 | 871 | 100.00% | yes | `9.664e-8` |
| 23 | 834 | 100.00% | yes | `1.208e-8` |

These groups reached float32 numerical saturation: all remaining proposals
failed the same-batch monotonicity test after 16 halvings. The preregistration
required 1,200 accepted updates, so the mechanical classification is
`calibrated_optimizer_pipeline_invalid` despite passing strength. Neural Stage
2 was not authorized and did not run.

The successful retained campaign took 28.50 seconds and 111.75 scheduled
process-seconds with a peak of four MLX processes. A report-field typo caused
four pre-artifact failures and 9.13 wasted process-seconds; the queue halted,
the fix was tested and synchronized, source identity was refrozen, and only
missing artifacts were rerun.

ADR 0108 subsequently repaired only the stop rule on the five saturated
groups, reused the remaining 19 groups unchanged, and recombined Stage 1 at
96.59% recall and 79.17% exact sets with every gate passing. Neural Stage 2 is
therefore authorized with this unchanged optimizer mechanism.

Machine-readable Stage 1 result:
`artifacts/experiments/complete-action-frontier-calibrated-monotone-adamw-v1/reports/free-combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-calibrated-monotone-adamw-v1-result.md`.
