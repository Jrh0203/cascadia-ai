# ADR 0090: Frontier Ranker Failure Classification

Status: complete; optimization or capacity underfit selected.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-failure-diagnostics-v1`

## Context

ADR 0089 proved that a deterministic frontier-plus-R1200 target can recover
99.58% of validation winners, but its selected learned ranker recovered only
26.21% of nonfrontier target slots and no complete target set. Four duplicate
training replicas agreed on the failure. The next cluster use must distinguish
why the learner failed before another treatment is trained.

## Hypotheses

Exactly one independent diagnostic runs per Mac:

| Host | Diagnostic | Question |
|---|---|---|
| john1 | train fit | Can the selected checkpoint fit its open training target? |
| john2 | observable collision | Do exact model-visible contexts require contradictory target labels? |
| john3 | objective gradient | Is target-set pressure dominated by or opposed to auxiliary losses? |
| john4 | error anatomy | Are misses concentrated in an observable, actionable slice? |

The four jobs share the selected ADR 0089 checkpoint and immutable train and
validation datasets. They are not replicas and do not train a model.

## Frozen Inputs

- Selected checkpoint:
  `step-000003592-epoch-0008-batch-000000` from the john2 ADR 0089 run.
- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- No sealed test, gameplay, new rollout, new label, K2048, cloud, or external
  compute.

Every report must carry the same model, checkpoint-manifest, train-manifest,
and validation-manifest BLAKE3 identities. Any mismatch invalidates the fork.

## Frozen Diagnostics

### Train Fit

Evaluate the selected checkpoint over every train and validation group. Call
optimization or capacity underfit material if train target-positive recall is
below 80% or exact target-set recovery is below 25%. Only if both train gates
pass may a train-to-validation target-recall gap of at least 10 percentage
points classify generalization failure.

### Exact Observable Collision

Fingerprint the exact parent tensors and permutation-invariant candidate
multiset consumed by the deployed model. Within each complete context, group
candidate rows by every candidate tensor consumed by the model. Count a
contradiction only when an exactly identical scored candidate is target
positive in one occurrence and target negative in another. Exact collision is
material only if contradictory occurrences contain at least 1% of all target
positive slots.

### Objective Gradient

Use the eight widest train groups, selected by descending candidate count and
deterministic shard/offset tie-breaks. At the selected checkpoint measure
scalar loss, full-parameter L2 gradient norm, and target/listwise cosine for:

- target-set cross entropy, weight 1.0;
- R1200 listwise loss, weight 0.5; and
- screen-only regularization, weight 0.01.

Target pressure is dominated if its weighted norm is below half the combined
weighted auxiliary norm. Direct conflict is material if target/listwise cosine
is at most -0.25 and auxiliary norm is at least half target norm. Conflict has
priority over domination.

### Error Anatomy

Score every validation target slot and decompose recall by phase, candidate
count, frontier count, action family, screen rank, drafted wildlife, and
immediate-delta profile. A slice is concentrated only with at least 50 target
positives, at least 35% of all misses, and recall at least 10 percentage points
below overall recall.

## Frozen Selection

Combine all four reports only after identity and sealed-domain checks pass.
Select the next mechanism in this order:

1. exact representation collision;
2. objective conflict;
3. objective domination;
4. optimization or capacity underfit;
5. generalization failure;
6. concentrated slice-specific representation;
7. diffuse model misspecification.

The result authorizes one single-host MLX pilot family only. It does not
authorize duplicate seeds. During that pilot the other three Macs continue
independent implementation, diagnostics, or CPU-heavy data work. Replication
opens only after the pilot passes its own frozen validation gate.

## Execution And Throughput

Each job runs under the host lock and `caffeinate`. All four launch
concurrently. Reports record wall time, peak RSS, process swaps, host, and
scientific digest. A healthy host may not sit idle while its assigned job is
ready. CPU percentage is reported, but accepted/rejected hypotheses per
wall-clock hour is the governing throughput metric.

## Maximum Compute

One execution of each frozen diagnostic, one deterministic combine pass, and
correctness/reporting work. No retraining, sweep, duplicate diagnostic,
threshold change, or domain opening is authorized.

## Result

All four diagnostics completed under host locks from one byte-identical
86-file MLX runtime bundle. Every report carried the same selected model,
checkpoint manifest, train manifest, and validation manifest identities. No
sealed data, gameplay, new labels, training, or external compute was used.

The train-fit audit was decisive. The selected checkpoint recovered only
29.36% of train target slots and exactly recovered 0.18% of train target
sets. Validation reached 26.21% and 0%, respectively. The target-recall gap
was only 3.15 percentage points, far below the 10-point generalization gate.
The learner therefore failed to fit its own open training target.

The other diagnostics excluded narrower explanations:

- exact model-visible collisions were zero across 800 contexts and 2,995,314
  actions;
- target/listwise gradients were strongly opposed at cosine `-0.908`, but the
  weighted target norm was `7.746` versus `1.695` combined auxiliary norm, so
  the frozen conflict and domination gates did not classify the auxiliary
  objective as primary;
- validation misses were broad across phase, wildlife, frontier count, and
  action family, with no slice meeting all concentration thresholds.

The preregistered selection therefore chooses
`optimization_or_capacity_underfit` and authorizes one single-host
target-set curriculum pilot with a capacity or optimizer change. Duplicate
training seeds remain closed until that pilot earns confirmation.

Machine-readable combined report:
`artifacts/experiments/complete-action-frontier-failure-diagnostics-v1/reports/combined.json`.
