# ADR 0121: Conditional Tile Generalization Forensics

Status: complete

Date: 2026-06-16

Experiment ID: `conditional-tile-generalization-forensics-v1`

## Context

ADR 0120 is the sole frozen optimizer-schedule treatment and runs on john2.
It cannot be altered or overridden by this audit. john1, john3, and john4 must
use the origin wall time for independent evidence rather than duplicate
training or wait for the checkpoint.

ADR 0118 established a large generalization split: 99.80% train recall and
67.75% validation recall under 200 fixed-rate epochs. Three mechanisms remain
scientifically distinct:

1. the pointwise model input maps identical observables to contradictory
   target-membership labels;
2. validation model inputs are materially outside the train distribution; or
3. late training expands normalized target margins only on train.

All arms use only the open ADR 0115 caches and frozen ADR 0116/0118 weights.
They perform no training, do not inspect sealed test or gameplay, and cannot
change ADR 0120's gates.

## Frozen Arms

### Exact Observable Aliasing, john3

For every tile item, hash the exact float32 bytes consumed by the deployed
pointwise ranker:

- group public state;
- conditional tile query context; and
- tile item features.

Measure repeated fingerprints, fingerprints observed with both target labels,
affected observations, affected positive observations, exact train/validation
overlap, and cross-split label contradictions.

Classify `observable_label_aliasing_material` when either:

- at least 1% of train positive observations belong to a fingerprint carrying
  both labels; or
- at least 1% of exact train/validation overlapping observations disagree in
  label.

Otherwise classify `observable_label_aliasing_not_material`.

### Input Distribution Shift, john4

For group state, tile query context, and tile item features, report exact train
and validation means, variances, minima, maxima, standardized mean
differences, and validation cells outside the train min/max support. Also
report tile-query width histograms and Jensen-Shannon divergence.

Classify `input_covariate_shift_material` when any frozen condition holds:

- tile-query width Jensen-Shannon divergence is at least 0.10;
- at least 10% of active dimensions in any input block have absolute
  standardized mean difference at least 0.50; or
- at least 1% of validation cells in any input block lie outside train
  min/max support.

Otherwise classify `input_covariate_shift_not_material`.

### Normalized Margin Specialization, john1

Replay the ADR 0116 20-epoch and ADR 0118 200-epoch checkpoints on train and
validation. For every tile query with both targets and nontargets, divide the
weakest-target minus strongest-nontarget boundary margin by within-query score
standard deviation. Report median and quantiles overall and by query-width
stratum.

Classify `late_fit_margin_specialization` only when the 200-epoch checkpoint:

- improves median normalized train margin by at least 0.50;
- improves median normalized validation margin by at most 0.10; and
- expands the train-minus-validation median margin gap by at least 0.50.

Otherwise classify `late_fit_margin_specialization_not_proven`.

## Pipeline Gates

Each arm must preserve cache and checkpoint identities, cover every required
query/item exactly once, remain finite, write an atomic report, and keep sealed
test, gameplay, teacher compute, cloud, Modal, and external compute closed.
Any violation invalidates only that arm.

## Decision Use

This audit does not authorize an ADR 0120 outcome. If ADR 0120 fails:

1. material exact aliases select a query-set-aware scorer that can condition
   each item on its sibling set;
2. otherwise material covariate shift selects a representation and
   distribution-robustness treatment;
3. otherwise proven margin specialization selects structural regularization
   rather than another learning-rate schedule; and
4. otherwise the successor must leave this pointwise tile-ranker family and
   test a separately preregistered query-set-aware complete-action mechanism.

If ADR 0120 passes, these results remain descriptive evidence for selector
design.

## Cluster Execution

- john2 continues the sole ADR 0120 origin.
- john3 owns exact observable aliasing.
- john4 owns input distribution shift.
- john1 owns normalized margin specialization, implementation, tests, and
  combined reporting.

The three arms are independent and launch concurrently. Duplicate training,
replicas, and idle reservations are prohibited.

## Maximum Compute

One complete open-cache pass per arm, two frozen checkpoint replays for the
margin arm, focused and full tests, one combined report, and documentation.
No training, new data, teacher rollout, sealed test, gameplay, cloud, Modal,
or external compute.

## Result

Every pipeline, cache-identity, checkpoint-identity, coverage, numerical, and
closed-domain gate passed.

- john3 classified `observable_label_aliasing_not_material`. All 850,246 train
  and 348,069 validation tile items had unique exact deployed-input
  fingerprints. There were zero within-split contradictions and zero exact
  cross-split overlaps.
- john4 classified `input_covariate_shift_not_material`. Query-width
  Jensen-Shannon divergence was 0.024838. The largest fraction of active
  dimensions above absolute SMD 0.50 was 4.55%, and the largest validation
  outside-support cell fraction was 0.0072%; every frozen threshold was
  missed.
- john1 classified `late_fit_margin_specialization`. From 20 to 200 fixed-rate
  epochs, median normalized train boundary margin improved by 1.7033 while
  validation worsened by 1.1260. The train-validation margin gap expanded by
  2.8293.

The frozen successor if ADR 0120 fails is `structural_regularization`.
Exposure, target-mass sampling, exact observable aliases, broad covariate
shift, and another learning-rate schedule are not authorized explanations.

Combined scientific BLAKE3:
`a1ee7130e3d086d79d6a55f6474f0ef4d73a80090f6c7eeecad0aecc52f9da09`.
