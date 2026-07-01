# ADR 0091: Frontier Target-Only Curriculum

Status: rejected on open validation; sealed test and gameplay closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-target-curriculum-v1`

## Context

ADR 0090 selected optimization or capacity underfit. The ADR 0089 checkpoint
recovered only 29.36% of train target slots and 0.18% of complete train target
sets, almost identical to validation. Exact observable collisions were zero.

A bounded-score reachability audit also excludes the existing ±12 residual
range as the structural limit. An optimistic ±3 correction recovers 99.87% of
train and 99.88% of validation target slots; ±6 recovers every target set.
The selected learner is using enough score range in principle but is not
optimizing the set allocation.

The remaining sharp mechanism is objective interference. On the eight widest
train groups, target and R1200-listwise gradients had cosine `-0.908`. Target
pressure was larger, so ADR 0090 did not classify conflict as the primary
failure, but the opposed auxiliary term can still prevent convergence when
the target-only checkpoint metric is not used.

## Hypothesis

Warm-starting the exact selected ADR 0089 checkpoint and fine-tuning only the
deployable target-set cross entropy, while selecting checkpoints directly on
validation target-slot miss rate, will materially fit the open training target
and transfer that improvement to validation.

## Frozen Treatment

- Host: john2 only.
- Seed: `2026061605`.
- Warm start:
  `step-000003592-epoch-0008-batch-000000` from ADR 0089.
- Architecture, observable inputs, ±12 residual range, anchored selector,
  proposal width, datasets, rotation augmentation, and packing remain
  unchanged.
- Loss: target-set cross entropy only.
- Optimizer: fresh AdamW at learning rate `3e-5`, weight decay `1e-4`.
- Maximum 20 epochs, six-epoch validation patience, checkpoints every 250
  optimizer steps.
- Checkpoint selection: lowest validation target-positive miss rate, then
  higher exact target-set fraction, then lower retained R4800 regret.

The target/listwise and screen-only auxiliary terms are removed. No
architecture, width, data, label, or score-range change is allowed.

## Pilot Gates

Evaluate the selected checkpoint over every train and validation action. The
pilot advances only if:

- train target-positive recall is at least 60%;
- train exact target-set recovery is at least 5%;
- validation target-positive recall is at least 50%;
- validation exact target-set recovery is at least 1%;
- validation exact R4800-winner recall does not fall below 75%;
- validation confidence-set coverage does not fall below 90%;
- retained validation regret remains below 0.15;
- every action is scored exactly once with finite scores;
- peak RSS remains below 4 GiB with zero process swaps; and
- sealed test and gameplay remain unopened.

Missing any gate rejects target-only fine-tuning. Passing authorizes one
confirmation seed and cross-host replay, not sealed test or gameplay.

## Cluster Use

john2 owns the only MLX training job. john1 coordinates and performs the final
train/validation audit. john3 owns the exact residual-reachability audit and
john4 owns an independent checkpoint/optimizer-trajectory audit while
training runs. No host duplicates the training seed.

## Maximum Compute

One 20-epoch target-only pilot, one open train/validation evaluation, the two
independent bounded diagnostics, and correctness/reporting work. No second
seed, architecture sweep, learning-rate sweep, new teacher compute, sealed
test, gameplay, K2048, cloud, or external compute is authorized.

## Result

The single john2 pilot stopped after seven epochs under the frozen six-epoch
futility rule. Epoch 1 was selected at checkpoint
`step-000000452-epoch-0001-batch-000000`. Its validation target-positive
recall was 26.29%, only 0.08 percentage point above the warm start, and it
recovered no complete validation target set.

The complete open-split replay measured:

- train target-positive recall `30.97%` versus the `60%` gate;
- train exact target-set recovery `0.18%` versus the `5%` gate;
- validation target-positive recall `26.29%` versus the `50%` gate;
- validation exact target-set recovery `0%` versus the `1%` gate;
- validation exact winner recall `74.58%` versus the `75%` floor;
- validation confidence coverage `90.00%`; and
- retained validation regret `0.065729`.

Five substantive pilot gates failed. Every integrity and finite-score gate
passed, peak evaluation RSS was 433 MB, process swaps were zero, and the
sealed test remained unopened.

Training loss fell from `6.0289` at epoch 1 to `5.8119` at epoch 7 while
validation target recall fell from `26.29%` to `24.41%`. The independent
trajectory audit showed the original four-loss run had the same failure mode,
never exceeding 26.03% target recall. The independent reachability audit
proved that ±6 residual capacity can recover every target set, so neither the
available score range nor auxiliary losses explain the remaining failure.

The hypothesis is rejected. Uniform target-set cross entropy is not calibrated
to exact top-K retention in this thousands-of-actions regime. The next
treatment must directly optimize the target/non-target cutoff, such as a
smooth worst-target versus strongest-nontarget boundary objective, while
keeping the same data and architecture.

Machine-readable evaluation:
`artifacts/experiments/complete-action-frontier-target-curriculum-v1/runs/john2-seed-2026061605/open-evaluation.json`.
