# ADR 0117: Conditional Tile Capacity and Query Audit

Status: complete; `full_data_scale_or_optimization_insufficient`

Date: 2026-06-16

Experiment ID: `conditional-tile-capacity-query-audit-v1`

## Context

ADR 0116 passed every pipeline gate but rejected boundary-only BCE. Removing
the ADR 0115 regression and listwise terms improved tile factor recall from
72.60% to 77.21% on train and from 66.57% to 70.59% on validation. It did not
repair complete-action proposal recall, which fell from 72.48% to 71.83%.

The remaining failure is primarily train underfit, not a validation-only gap.
Exact model-visible tile inputs have no target-label collisions, and the
conditional hierarchy has a 99.18% validation oracle ceiling. The next
question is therefore whether the unchanged 256-wide set ranker can fit hard
queries locally, whether fit collapses as independent queries share
parameters, or whether explicit candidate-to-candidate interaction is
required.

## Frozen Evidence

- ADR 0116 classification:
  `target_only_tile_objective_insufficient`.
- ADR 0116 combined scientific BLAKE3:
  `0d4681df11a527ca1571008cca8b6e55800866380fb8a4cf7450def1ed54a4f6`.
- Target-only tile weights BLAKE3:
  `5c13fe87d7b4ac0a8ff9f647f57c69b8d9ab583b3ce2e85e41ee0f3d97e8f514`.
- Train cache payload BLAKE3:
  `1707fd84fac77dee0e4878165bf8f8b98869b6d4d206deb55db030321cc96ede`.
- Validation cache payload BLAKE3:
  `b128a3b5bf53e135febf39dba02d9c7486692245523516a5ee3031eea795229b`.

Sealed test, gameplay, new teacher compute, cloud, and external compute remain
closed.

## Frozen Query Cohorts

Only open train queries wider than 32 with at least one target and one
nontarget are eligible. Rank eligible queries by BLAKE3 of:

`train-cache-payload : 2026061649 : shard-index : query-index`

The first 16 queries are the small cohort. The first 256 queries are the
medium cohort, so the small cohort is an exact subset. Selection does not use
model scores, validation, or outcomes.

## Four Distinct Arms

1. `anatomy`, john1
   - Compare frozen ADR 0115 and ADR 0116 checkpoints on full open train and
     validation.
   - Report exact width-bin and phase recall, exact sets, and target/nontarget
     margins.
   - Perturb only the ADR 0116 checkpoint with permuted or zero query context,
     permuted state, and zeroed tile-factor, local-geometry, or descendant
     item blocks.
   - No gradients or training.
2. `baseline-16`, john2
   - Train the unchanged ADR 0116 `HierarchicalFactorRanker` from scratch on
     the 16-query cohort.
   - Balanced target-membership BCE only, AdamW `3e-4`, weight decay `1e-4`,
     batch 16, seed `2026061649`, at most 2,000 updates.
3. `baseline-256`, john3
   - The same unchanged ranker, objective, optimizer, seed, and data
     construction on the 256-query cohort.
   - Batch 16, at most 4,000 updates.
4. `attention-256`, john4
   - The same inputs, objective, optimizer, seed, cohort, and update budget as
     `baseline-256`.
   - Replace only mean/max-only set interaction with two 8-head, 256-wide
     pre-norm candidate self-attention blocks before the unchanged pooled
     output head.

Training arms evaluate only their own open train cohorts. They never inspect
validation for selection or reporting. Evaluate every 100 updates and stop
only after three consecutive 100% exact-query checkpoints or the maximum
budget. Save the best train checkpoint by recall, exact recovery, then loss.

## Pipeline Gates

Every arm must:

- use the exact frozen implementation and cache identities;
- cover its selected queries and items exactly once per evaluation;
- remain finite, below 4 GiB peak process RSS, and at zero process swaps;
- preserve the nested 16/256 query selection exactly across hosts;
- avoid duplicate discovery work; and
- keep sealed test, gameplay, validation-driven selection, new teacher
  compute, cloud, and external compute closed.

## Mechanical Classification

Apply pipeline validity before scientific strength.

1. `conditional_tile_capacity_audit_invalid`
   - any identity, coverage, numerical, resource, cohort, or sealed-domain
     gate fails.
2. `local_baseline_fit_insufficient`
   - `baseline-16` best recall is below 99.5% or exact recovery below 95%.
3. `full_data_scale_or_optimization_insufficient`
   - the small arm passes and `baseline-256` reaches at least 98% recall and
     90% exact recovery.
4. `query_relational_representation_insufficient`
   - `baseline-256` misses its gate, while `attention-256` reaches at least
     98% recall and 90% exact recovery and improves recall by at least 5
     points and exact recovery by at least 10 points.
5. `shared_capacity_or_optimization_insufficient`
   - the small arm passes, the medium baseline fails, and the attention
     control does not earn the relational classification.

The anatomy arm is explanatory rather than a strength override. A validation
recall drop of at least one point under permuted query context is considered
material use of that context.

Only the mechanically selected mechanism may authorize the next single
treatment. This audit does not authorize a full policy/value trainer, sealed
test, gameplay, a width sweep, or duplicate seeds.

## Cluster Execution

Launch all four distinct arms concurrently. Each Mac owns one scientific
question, not a replica. When an arm finishes, its host immediately performs
cross-host artifact transfer, integrity checks, tests, report work, or the next
dependency-ready task. No host waits while compatible queued work exists.

The combined report records arm wall time, per-host work, duplicate-compute
fraction, and the mechanism decision reached per campaign hour.

## Maximum Compute

One full frozen anatomy pass, one 16-query baseline origin, one 256-query
baseline origin, one 256-query attention origin, integrity checks, focused and
full tests, one combined report, and documentation. No second seed,
confirmation replica, hyperparameter sweep, new data, teacher rollout, sealed
test, gameplay, cloud, Modal, or external compute.

## Result

Every pipeline gate passed. The nested cohorts matched exactly across hosts,
all four arms remained finite below 2 GiB peak process RSS with zero process
swaps, and sealed test, gameplay, validation-driven selection, new teacher
compute, cloud, and external compute remained closed.

The unchanged baseline ranker reached:

- 100% recall and 100% exact recovery on 16 hard queries after 200 updates,
  sustained through the preregistered three-check stop at update 400; and
- 100% recall and 100% exact recovery on 256 hard queries at update 3,200,
  sustained through the stop at update 3,400.

The attention control reached 99.95% recall and 98.83% exact recovery only at
its full 4,000-update budget. It added 1.58 million parameters, trained more
slowly, and did not beat the unchanged baseline.

The frozen anatomy found that permuting query context reduced validation
recall by only 0.59 points and permuting parent state changed it by 0.00
points. Zeroing descendant summaries reduced validation recall by 27.47
points. The selected model therefore relies overwhelmingly on
candidate-conditioned descendant evidence and does not currently gain material
validation value from explicit parent or draft context.

The mechanical classification is
`full_data_scale_or_optimization_insufficient`. The model and target-only
objective can fit representative hard queries exactly. ADR 0116's full-cache
trajectory was still improving at epoch 20, from 74.18% recall at epoch 10 to
77.21% at epoch 20, while the medium cohort required roughly 200 passes for
exact fit. The next treatment must change only full-cache training exposure,
not architecture, features, objective, or width.

The four-host campaign resolved four distinct questions in 264.50 seconds,
scheduled 454.12 process-seconds, used zero duplicate discovery compute, and
reached 54.44 decisions per wall-clock hour.

Machine-readable combined scientific BLAKE3:
`695d4bce6a82047ff73e46a740ae1ef302a6995b9ca6dc14ae82895a22333eae`.
