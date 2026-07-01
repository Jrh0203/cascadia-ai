# ADR 0103: Frontier Free-Residual Objective And Optimizer Audit

Status: completed as `free_residual_pipeline_invalid`; sealed test and
gameplay closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-free-residual-audit-v1`

## Context

ADR 0102 classified scale-16 expected-rank failure as
`local_optimization_or_representation_insufficient`. One-group and four-group
training failed local recovery, independent selected-checkpoint adaptation
reached only 40.66% recall with zero exact sets, width scaling was immaterial,
and geometric gradient conflict did not produce empirical interference.

The current objective, optimizer, bounded residual parameterization, and full
model are still confounded. Before another trainer or architecture treatment,
this audit asks:

1. Does the exact scale-16 cross-entropy optimum inside the deployed residual
   box recover the deployed top-64 selector?
2. Can frozen AdamW reach that optimum when every action has an independent
   residual parameter?
3. Does an independent projected convex optimizer reproduce the analytic
   optimum?
4. Does extending full-model local adaptation from 120 to 1,200 exposures
   recover the first four groups?

## Frozen Inputs

- The exact ADR 0102 train dataset, scale-16 cache, selected ADR 0101
  checkpoint, ordered 64-group cohort, and cohort digest.
- Cohort digest:
  `30899dec701f053d96023f963b473681516fb0df00a58edf54146c623fd2769d`.
- Selected model BLAKE3:
  `5b50a1db5f1f415ad6a10a7588d9521d6c11a9408be2e67d5691e85f60c04869`.
- Target scale 16, student temperature 2, frontier-anchored width 64,
  residual range plus or minus 12, and stable action-hash tie break.
- AdamW learning rate `1e-4`, weight decay `1e-4`, seed `2026061631`.

No validation treatment selection, sealed test, gameplay, new teacher
compute, cloud, Modal, or external compute is allowed.

## Arm A: Exact Box-Constrained Objective Optimum

Host: john1.

For every one of the 64 ordered cohort groups, minimize the exact deployed
scale-16 cross entropy over one independent score per eligible action with:

`screen_value - 12 <= score <= screen_value + 12`.

Use the convex KKT form
`score_i = clip(T * log(p_i) + c, lower_i, upper_i)` for positive target mass,
place zero-mass eligible actions at their lower bound, and solve the unique
normalization scalar `c` by deterministic bisection. Report objective,
top-64 recovery, active lower/interior/upper counts, maximum KKT violation,
and finite-score coverage per group.

The analytic optimum passes only with at least 95% aggregate target recall,
75% exact sets, maximum KKT violation at most `1e-8`, and complete finite
coverage.

## Arm B: Frozen AdamW With Free Per-Action Residuals

Host: john2.

Use the first 24 cohort groups. For each group independently, initialize one
raw residual parameter per action from the exact selected-checkpoint residual:

`raw_i = atanh(clamp(residual_i / 12, -0.999999, 0.999999))`.

Optimize the unchanged scale-16 objective through
`score_i = screen_i + 12 * tanh(raw_i)` using AdamW `1e-4`, weight decay
`1e-4`. Evaluate at 0, 6, 24, 60, 120, 300, 600, and 1,200 updates. No
rotation is needed because the parameter is attached to the exact stable
action order.

Frozen free-residual optimization passes at 1,200 updates with at least 95%
aggregate recall and 75% exact sets. The 120-update point is retained to
measure whether ADR 0102's local budget was sufficient.

## Arm C: Independent Projected Convex Control

Host: john3.

Use the same first 24 groups and selected-checkpoint scores. Optimize scores
directly inside the residual box with deterministic accelerated projected
gradient descent, monotone restart/backtracking, and the exact analytic
gradient. Start at step size 8, halve until the objective does not increase,
stop at projected KKT residual `1e-9` or 10,000 iterations, and report the
full trajectory.

This arm passes only when every group converges, maximum KKT residual is at
most `1e-8`, objective differs from Arm A by at most `1e-7`, and target
selection metrics match Arm A exactly on the shared 24 groups.

## Arm D: Full-Model Long-Horizon Local Continuation

Initial host: john4. After Arms A-C finish, john1-john3 backfill disjoint
group shards so all four first groups run concurrently.

For each of the first four groups independently, reset to the selected
checkpoint and continue the complete model with the unchanged AdamW settings
for 1,200 exposures. Cycle rotations exactly `0,1,2,3,4,5`; evaluate at 0,
120, 300, 600, and 1,200 exposures.

Long-horizon full-model local recovery passes with at least 90% aggregate
recall and 75% exact sets at 1,200 exposures. The four shards must have
disjoint group IDs and combine in frozen cohort order.

## Selector Ceiling

Every Arm A group also evaluates the known box-feasible selector assignment:
target actions at `screen + 12`, eligible nontargets at `screen - 12`.
It must recover 100% of target positives and 100% of target sets. Failure
invalidates the pipeline rather than selecting an objective treatment.

## Pipeline Gates

Every arm must prove exact dataset, cache, checkpoint, cohort, and source
identity; complete finite coverage; peak RSS at most 4 GiB; zero process
swaps; no attributable positive system-swap growth; and closed test,
gameplay, teacher, and external-compute domains. Every origin report receives
one cross-host scientific replay.

## Mechanical Classification

Apply precedence in this order:

1. `free_residual_pipeline_invalid`
   - any pipeline, selector-ceiling, projected-control, or replay gate fails.
2. `scale16_objective_box_misaligned`
   - the selector ceiling passes but the exact analytic optimum fails its
     selection gate.
3. `frozen_optimizer_hyperparameters_insufficient`
   - the analytic optimum and projected control pass but frozen free-residual
     AdamW fails at 1,200 updates.
4. `full_model_local_budget_insufficient`
   - free-residual AdamW passes, full-model recovery fails at 120 exposures,
     and passes at 1,200 exposures.
5. `public_observable_representation_insufficient`
   - the analytic optimum, projected control, and free-residual AdamW pass,
     but full-model recovery fails at 1,200 exposures.
6. `local_failure_not_reproduced`
   - full-model recovery already passes at 120 exposures.
7. `local_mechanism_unresolved`
   - no classification above applies.

The selected result authorizes exactly one successor mechanism: objective,
optimizer, training budget, representation, or another bounded diagnostic.

## Cluster And Throughput Contract

Launch Arms A-D as four distinct first-wave jobs. When A-C finish, their hosts
immediately backfill the three remaining disjoint Arm D group shards. Do not
duplicate a discovery shard. Replays are a separate confirmation wave. Record
critical path, scheduled host-seconds, per-host work, confirmation fraction,
and any avoidable idle interval while compatible work was queued.

## Maximum Compute

Exactly 64 analytic groups, 24 frozen-Adam groups, 24 projected-control
groups, four disjoint 1,200-exposure neural groups, one cross-host replay per
arm, tests, source checks, and report generation. No extra group, seed,
optimizer sweep, learning rate, full trainer, validation treatment, sealed
test, or gameplay run.

## Result

Every authorized origin report and cross-host replay completed. All seven
scientific payloads reproduced bit-identically, source identity matched across
john1-john4, every process remained below 1 GiB RSS with zero process swaps,
and all sealed domains remained closed.

Arm A proved that the objective and residual box are sound. The exact
box-constrained scale-16 optimum recovered 100% of 2,230 target slots and
100% of 64 target sets, with maximum KKT violation `1.471e-15`. The explicit
selector ceiling also recovered 100%.

Arm B showed that frozen AdamW is extremely slow even after representation is
removed. One free raw residual per action reached only 39.72% recall after 120
updates and 59.22% after 1,200 updates, with zero exact sets.

Arm D showed that a tenfold neural local budget is also insufficient on the
first four groups. Aggregate recall rose from 38.73% at 120 exposures to
58.45% at 1,200 exposures, but all four groups retained zero exact recovery.

Arm C narrowly missed its preregistered numerical-control gate. Accelerated
projected optimization reached 96.47% recall and 79.17% exact sets, but its
maximum KKT violation was `3.304e-8` against `1e-8`, and its maximum objective
gap was `2.622e-7` against `1e-7`. The replay was bit-identical. Under the
frozen precedence, the mechanical classification is therefore
`free_residual_pipeline_invalid`; the strong optimizer evidence cannot yet
authorize a treatment.

The four distinct first-wave jobs and three disjoint neural backfill shards
completed in 241.21 seconds. The complete origin-plus-confirmation campaign
took 487.70 seconds and 1,205.11 scheduled host-seconds. Confirmation consumed
49.81% of scheduled compute; duplicate discovery remained zero.

The only authorized successor is a preregistered repair of the failed
projected numerical-control gate. It may reuse the frozen analytic,
free-AdamW, and neural evidence, but may not change or launch a model
treatment.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-free-residual-audit-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-free-residual-audit-v1-result.md`.
