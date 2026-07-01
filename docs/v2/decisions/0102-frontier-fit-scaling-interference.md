# ADR 0102: Frontier Fit Scaling And Cross-Group Interference

Status: completed as `local_optimization_or_representation_insufficient`;
sealed test and gameplay closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-fit-interference-audit-v1`

## Context

ADR 0101 concentrated 93.76% of the expected-rank target mass inside the
deployed nonfrontier set and raised the initial in-set gradient share to about
48%. The selected full-dataset model nevertheless recovered only 30.23% of
train targets and 0.18% of exact train sets. Its train fit was worse than ADR
0100, while independent cache, gradient, reachability, replay, memory, and
finite-score checks passed.

The next question is not another loss, temperature, seed, warm start, or full
trainer. It is whether the unchanged public-observable model fails because:

1. even one or a few groups cannot be fit by local optimization;
2. fit collapses as independently fit-able groups share finite capacity;
3. gradients from different groups destructively interfere; or
4. finite capacity and destructive interference are both material.

This ADR resolves that question on existing open train data before authorizing
another model treatment.

## Frozen Inputs

- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Train manifest BLAKE3:
  `7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99`.
- Canonical ADR 0101 scale-16 train cache:
  `artifacts/experiments/complete-action-frontier-expected-rank-scale16-v1/cache/john2/train`.
- Selected ADR 0101 checkpoint:
  `step-000004514-epoch-0010-batch-000000`.
- Selected model BLAKE3:
  `5b50a1db5f1f415ad6a10a7588d9521d6c11a9408be2e67d5691e85f60c04869`.
- Target scale 16, student temperature 2, anchored width 64, residual range
  plus or minus 12, stable action-hash tie break, and the exact ADR 0101 loss.
- AdamW learning rate `1e-4`, weight decay `1e-4`.
- Seed `2026061630`.

No new teacher samples, validation-driven treatment selection, sealed test,
gameplay, cloud, Modal, or external compute is allowed.

## Deterministic Cohort

Enumerate every train group with its group ID, phase, candidate count, shard,
and group reference. Assign width buckets `at_most_2048`, `2049_to_4096`, and
`above_4096`. Within each phase-by-width cell, order groups by BLAKE3 of:

`experiment_id || little_endian_u64(group_id)`

Interleave the nine ordered cells in lexical cell order until all groups are
exhausted. The first 64 groups are the audit cohort. Every arm writes the exact
ordered group IDs and cohort digest. All cross-arm shared prefixes must match
byte for byte.

Training uses one decision group per optimizer step. An exposure is one update
to one group. Rotations cycle exactly through `0,1,2,3,4,5`; therefore every
multiple of six exposures gives each group uniform rotation exposure. Within
each six-exposure block, group order is deterministically shuffled from the
frozen seed and arm identity. All fit comparisons give every included group
the same number of exposures.

## Arm A: Nested-Subset Memorization

Host: john1.

Train separate default-width models from the identical frozen initialization
on nested cohort prefixes of 1, 4, 16, and 64 groups. Give every group 60
updates, ten at each exact rotation. Evaluate the complete training subset
after 6, 12, 24, 36, 48, and 60 exposures per group.

Report target-positive recall, exact target-set fraction, mean objective,
R4800 winner retention, finite scores, elapsed time, RSS, and swaps for every
curve point.

Local-fit gates:

- size 1: recall at least 95% and exact fraction 100%;
- size 4: recall at least 90% and exact fraction at least 75%.

Scaling collapse is material when the size-64 final recall is at least 15
percentage points below size 4 or its exact fraction is at least 25 points
below size 4, and size-64 recall remains below 80%.

## Arm B: Capacity Scaling

Host: john2.

On the identical first 32 cohort groups, train three from-scratch models from
the frozen seed:

| Variant | Hidden width | Heads | Parameter scale |
|---|---:|---:|---|
| small | 96 | 6 | measured in report |
| baseline | 192 | 6 | measured in report |
| large | 288 | 6 | measured in report |

Board depth, market depth, feed-forward multiplier, heads, loss, optimizer,
rotation schedule, residual range, and every input remain unchanged. Give each
group 60 updates and evaluate at the same exposure checkpoints as Arm A.

Capacity is material only if:

- final recall is monotonic within a two-point tolerance from 96 to 192 to
  288;
- width 288 improves recall over width 192 by at least 8 points; and
- width 288 improves exact-set fraction over width 192 by at least 10 points.

All variants must remain below 4 GiB peak RSS with zero process swaps and no
positive system-swap consumption attributable to the process.

## Arm C: Gradient Conflict

Host: john3.

Use the first 32 cohort groups. At both the common from-scratch initialization
and the exact selected ADR 0101 checkpoint, compute one full-model float32
expected-rank gradient per unrotated group. Normalize only for cosine
calculation; do not clip, average, or update the model.

Report exact pairwise cosine distributions and each group gradient's cosine
against the sum of all other group gradients for:

- the full model;
- `residual_head`;
- `output_trunk`;
- `candidate_projection`; and
- all remaining representation parameters.

The scientific report stores the complete 32 by 32 cosine matrices, ordered
group IDs, gradient norms, and aggregate summaries. Intermediate gradient
vectors may be discarded after the report is atomically complete.

Destructive gradient interference is material only when, at the selected
checkpoint:

- at least 30% of full-model gradients have negative cosine against the sum of
  the other gradients;
- the median cosine against the other-gradient sum is at most `-0.02`; and
- at least 20% of off-diagonal full-model pairs have cosine at most `-0.10`.

Initialization geometry is diagnostic and cannot alone pass this gate.

## Arm D: Error Anatomy And Empirical Interference

Host: john4.

Use the first 24 cohort groups and the exact selected ADR 0101 model.

1. Independently reset to the selected checkpoint for each group and adapt the
   full model for 120 updates on that group, twenty per exact rotation.
2. Separately reset once to the selected checkpoint and adapt one shared model
   on all 24 groups for 120 updates per group with the same per-group rotation
   exposure.

Report every group's initial, independent, and shared recall, exact recovery,
objective, phase, width, candidate count, and recovery trajectory. Also report
aggregate and phase/width slices.

Independent local recovery passes at 90% aggregate recall and 75% exact sets.
Empirical interference is material only when independent adaptation exceeds
shared adaptation by at least 15 recall points and 25 exact-set points.

## Pipeline Gates

Every arm must prove:

- exact train manifest and scale-16 cache identity;
- identical ordered cohort prefix and cohort digest where shared;
- exact source bundle identity on john1-john4;
- complete finite scoring and gradients;
- all scheduled groups and exposure checkpoints present exactly once;
- peak process RSS at most 4 GiB;
- zero process swaps;
- no attributable positive system-swap consumption;
- sealed test and gameplay unopened;
- no new teacher or external compute.

Any failure classifies the campaign as `fit_interference_pipeline_invalid`.

## Mechanical Classification

Apply precedence in this order:

1. `fit_interference_pipeline_invalid`
   - any pipeline gate fails.
2. `local_optimization_or_representation_insufficient`
   - either nested local-fit gate fails; or
   - Arm D independent local recovery fails.
3. `mixed_capacity_and_interference`
   - scaling collapse is material;
   - the capacity gate passes; and
   - both gradient and empirical interference gates pass.
4. `shared_capacity_bottleneck`
   - scaling collapse is material;
   - the capacity gate passes; and
   - both interference gates do not pass.
5. `cross_group_gradient_interference`
   - scaling collapse is material;
   - both gradient and empirical interference gates pass; and
   - the capacity gate fails.
6. `shared_model_scaling_failure_unresolved`
   - scaling collapse is material but none of the discriminating combinations
     above pass.
7. `no_material_fit_scaling_failure`
   - local recovery passes and scaling collapse is not material.

The selected classification authorizes exactly one next mechanism:

- local insufficiency: a representation/local-optimizer mechanism;
- capacity: a larger-capacity architecture with the smallest sufficient width;
- interference: gradient-conflict mitigation without increasing width;
- mixed: the smallest passing width plus conflict mitigation;
- unresolved: another bounded diagnostic, not a full trainer;
- no scaling failure: re-audit the full-dataset training path.

No successor may alter more than the mechanism selected here.

## Cluster And Throughput Contract

- john1: nested-subset memorization and campaign coordination.
- john2: capacity scaling.
- john3: gradient-conflict geometry plus source/integrity backfill.
- john4: error anatomy and empirical interference.

These are four distinct scientific decisions, not replicas. Each arm uses a
separate host lock, artifact root, event log, atomic report, and stop rule.
When an arm finishes, its host backfills source verification, report replay,
or pending tests rather than duplicating another arm. The campaign closeout
reports productive wall time, idle time with compatible work queued, work per
node, useful CPU/MLX occupancy, duplicate compute fraction, critical-path
time, and frozen decisions per campaign hour.

## Maximum Compute

Exactly the four arms above, one source replay per report, focused and full
test suites, and report generation. No extra cohort, width, seed, exposure
budget, full 560-group trainer, validation treatment selection, sealed test,
or gameplay run.

## Result

All four preregistered arms completed, and every scientific payload reproduced
bit-identically on a second Mac.

The local-fit gates failed decisively:

- one-group recall was 18.92% with zero exact recovery;
- four-group recall was 30.28% with zero exact recovery; and
- independent 24-group selected-checkpoint adaptation reached 40.66% recall
  with zero exact recovery.

Shared scale was not the limiting mechanism. Recall rose from 30.28% at four
groups to 36.68% at 64 groups, so the frozen scaling-collapse gate failed.
Increasing hidden width from 96 to 192 to 288 produced 37.46%, 36.93%, and
38.78% recall with zero exact sets, so capacity was not material.

Gradient conflict was real but not sufficient to classify the failure.
At the selected checkpoint, 78.12% of full-model group gradients opposed the
sum of the other gradients, the median cosine was -0.128186, and 44.15% of
off-diagonal pairs were at most -0.10. However, independent adaptation beat
shared adaptation by only 2.23 recall points and zero exact-set points, far
below the empirical-interference gate. Because local recovery has precedence,
the mechanical classification is
`local_optimization_or_representation_insufficient`.

The first wave resolved four distinct hypotheses in 975.30 seconds with zero
duplicate discovery training. The complete origin-plus-replay campaign took
1,961.49 seconds and 5,260.36 scheduled host-seconds. Replays consumed 50.19%
of scheduled compute and were exclusively the preregistered confirmation
wave. All reports remained below 1.31 GiB RSS with zero process swaps.

The john1 nested origin observed unrelated positive system-wide swap growth
while recording zero process swaps. Its john4 replay was scientifically
identical with zero system-swap growth and is the pipeline-selected nested
report. The complete pipeline passed.

The next authorized experiment is one bounded local objective/optimizer audit.
It must compare the exact box-constrained free-residual optimum of the current
scale-16 loss with direct free-residual optimization and the deployed top-64
selector. It may not launch another full 560-group trainer, width treatment,
conflict-only treatment, sealed test, or gameplay evaluation.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-fit-interference-audit-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-fit-interference-audit-v1-result.md`.
