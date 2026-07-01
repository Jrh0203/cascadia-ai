# ADR 0092: Frontier Top-K Boundary Ranking

Status: rejected on open validation; sealed test and gameplay closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-boundary-ranking-v1`

## Context

ADR 0091 removed every auxiliary loss yet selected only 30.97% train and
26.29% validation target-slot recall. Its uniform set cross entropy continued
to fall while target recall deteriorated. Independent evidence ruled out
observable collisions and score-range insufficiency: an optimistic ±6
residual recovers every target set, versus the model's frozen ±12 range.

The deployment contract is not average target probability. The anchored
selector succeeds only when every required nonfrontier target needed to fill
width 64 outranks the strongest excluded nontarget. ADR 0092 therefore changes
only the objective and optimizes this exact cutoff directly.

## Hypothesis

A smooth weakest-target versus strongest-nontarget boundary loss will fit the
open training target materially better than uniform set cross entropy and
transfer enough of that improvement to validation to justify one confirmation
seed.

## Frozen Treatment

- Trainer: john2 only.
- Seed: `2026061606`.
- Warm start: the exact ADR 0089 john2 checkpoint
  `step-000003592-epoch-0008-batch-000000`, not ADR 0091.
- Architecture, observable inputs, ±12 residual range, anchored selector,
  width 64, datasets, augmentation, packing, and model initialization remain
  unchanged.
- Loss:
  - smooth target floor
    `-tau * logsumexp(-target_score / tau)`;
  - smooth nontarget ceiling
    `tau * logsumexp(nontarget_score / tau)`;
  - `tau * softplus((ceiling - floor + margin) / tau)`;
  - temperature `tau = 0.25`;
  - margin `0.5`.
- The log-sum-exp terms are deliberately not divided by set cardinality. The
  resulting floor is no greater than the weakest target and the ceiling is no
  less than the strongest nontarget, making the surrogate conservative.
- Fresh AdamW, learning rate `3e-5`, weight decay `1e-4`.
- Maximum 20 epochs, six-epoch validation patience, checkpoints every 250
  optimizer steps.
- Selection: lowest validation target-positive miss rate, then higher exact
  target-set fraction, then lower retained R4800 regret.

No listwise, cross-entropy, or screen regularization auxiliary is present.
No architecture, score range, data, labels, width, or selector change is
allowed.

## Pretraining Gates

Before training:

- the full Python suite and Ruff pass;
- john3 runs one full-model forward/backward/AdamW update on the widest open
  group;
- every target score gradient is negative, every eligible nontarget gradient
  is positive, and excluded actions have zero score gradient;
- the widest update has finite parameters, gradients, loss, and output;
- john4 directly optimizes the same loss on the 12 widest validation groups
  while clipping residuals to ±12;
- the bounded score-space audit reaches at least 99% target-slot recall and
  90% exact target sets;
- both audits stay below 4 GiB RSS with zero process swaps; and
- all source, dataset, and warm-start identities match the manifest.

Any failure rejects the launch before training.

## Pilot Gates

The selected checkpoint advances only if:

- train target-positive recall is at least 60%;
- train exact target-set recovery is at least 5%;
- validation target-positive recall is at least 50%;
- validation exact target-set recovery is at least 1%;
- validation exact R4800-winner recall is at least 75%;
- validation confidence-set coverage is at least 90%;
- retained validation regret is below 0.15;
- every train and validation action is scored exactly once with finite scores;
- evaluation stays below 4 GiB RSS with zero process swaps; and
- sealed test and gameplay remain unopened.

Passing authorizes one confirmation seed and cross-host replay. It does not
authorize sealed test, gameplay, new teacher compute, or a larger experiment.

## Throughput-First Cluster Use

- john1: protocol ownership, source identity, coordination, and final open
  evaluation.
- john2: the only MLX training job.
- john3: independent maximum-width gradient and optimizer-step audit.
- john4: independent bounded score-space convergence audit.

This is intentionally not four duplicate training replicas. Each Mac answers
a different question concurrently, maximizing validated research progress per
wall-clock hour. A second seed or host replica is allowed only after the pilot
passes and the scientific question becomes variance or portability.

## Maximum Compute

One 20-epoch pilot, one open train/validation evaluation, the two independent
support audits, and correctness/reporting work. No sweep, second training seed,
new teacher compute, sealed test, gameplay, K2048, cloud, or external compute.

## Result

Both launch audits passed. john3 exercised the widest 10,854-action group:
all target score gradients were negative, all 10,790 eligible nontarget
gradients were positive, excluded gradients were zero, and the full AdamW
update remained finite at 424 MB RSS with zero swaps. john4 then optimized
the same objective directly on the 12 widest validation groups inside the
±12 residual bound. Exact target-set recovery rose from 0% to 100%, target
recall rose from 36.41% to 100%, and the largest residual used was 9.403.

The neural pilot failed despite that reachable ceiling. None of six epochs
beat the untouched warm start, so checkpoint
`step-000000000-epoch-0000-batch-000000` remained selected. The complete open
evaluation reproduced the warm-start metrics:

- train target recall 29.36% versus the 60% gate;
- train exact sets 0.18% versus the 5% gate;
- validation target recall 26.21% versus the 50% gate;
- validation exact sets 0% versus the 1% gate;
- validation winner recall 76.67%;
- validation confidence coverage 90.42%; and
- validation regret 0.061734.

Four substantive gates failed. Training boundary loss nevertheless fell from
3.2696 in epoch 1 to 3.0191 in epoch 6 while validation target recall fell to
18.76%, winner recall to 57.92%, and confidence coverage to 81.25%. The
surrogate is therefore not merely hard to optimize; its concentrated smooth
minimum/maximum pressure is misaligned with the shared neural scorer.

The hypothesis is rejected. The next authorized objective must distribute
gradient across the full ordered boundary: pair each target rank from weakest
to strongest with a corresponding hard nontarget rank from strongest to
weakest, rather than reducing each set to one log-sum-exp extreme.
