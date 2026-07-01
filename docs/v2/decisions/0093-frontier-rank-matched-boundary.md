# ADR 0093: Frontier Rank-Matched Full Boundary

Status: rejected on open validation; sealed test and gameplay closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-rank-boundary-v1`

## Context

ADR 0092 proved that the open target is reachable inside the frozen ±12
residual range, but its smooth minimum/maximum surrogate was misaligned with
the shared model. Direct score optimization recovered every one of the 12
widest validation target sets. Neural loss then fell every epoch while
validation target recall collapsed from 26.21% to 18.76%.

The failure mechanism is concentrated gradient. One log-sum-exp target floor
and one nontarget ceiling reduce a width-32 allocation problem to two soft
extremes. Most required target slots receive little direct pressure, while
model updates coupled through shared features can improve the extremes yet
damage the deployed ordering.

## Hypothesis

Applying a margin loss to every ordered target/hard-negative rank will
distribute gradient over the complete nonfrontier fill boundary and improve
both train and validation target recovery without changing representation,
data, score range, or selection.

## Frozen Treatment

- Trainer: john2 only.
- Seed: `2026061607`.
- Warm start: ADR 0089 john2 checkpoint
  `step-000003592-epoch-0008-batch-000000`.
- Architecture, observable inputs, ±12 residual range, anchored selector,
  width 64, datasets, augmentation, packing, and initialization remain fixed.
- For each group:
  1. order target scores from weakest to strongest;
  2. order the same number of eligible nontarget scores from strongest to
     weakest;
  3. pair equal ranks;
  4. average
     `temperature * softplus((nontarget - target + margin) / temperature)`
     across all occupied pairs.
- Temperature: `1.0`.
- Margin: `0.5`.
- Maximum target/hard-negative pairs: `64`; the observed deployable quota
  determines the active pair mask.
- Fresh AdamW, learning rate `3e-5`, weight decay `1e-4`.
- Maximum 20 epochs, six-epoch validation patience, checkpoints every 250
  optimizer steps.
- Checkpoint selection: lowest validation target-positive miss rate, then
  higher exact target-set fraction, then lower retained R4800 regret.

No auxiliary loss, architecture change, data change, score-range change,
teacher compute, or width change is allowed.

## Pretraining Gates

- Full Python suite and Ruff pass.
- john3 runs one widest-group full-model update.
- Every target receives a strictly negative score gradient.
- Exactly one hardest nontarget per target receives a positive gradient; all
  remaining nontarget and excluded gradients are zero.
- Loss, gradients, updated parameters, and outputs remain finite below 4 GiB
  RSS with zero swaps.
- john4 directly optimizes the same objective on the 12 widest validation
  groups inside ±12.
- Direct target recall reaches at least 99% and exact target sets at least 90%.
- All source, dataset, and checkpoint identities match the locked manifest.

Any failure rejects the training launch.

## Pilot Gates

Use the unchanged ADR 0091/0092 gates:

- train target recall at least 60%;
- train exact target sets at least 5%;
- validation target recall at least 50%;
- validation exact target sets at least 1%;
- validation winner recall at least 75%;
- validation confidence coverage at least 90%;
- validation regret below 0.15;
- complete finite scoring, RSS below 4 GiB, zero swaps; and
- sealed test and gameplay unopened.

Passing authorizes one confirmation seed and cross-host replay only.

## Throughput-First Cluster Use

- john1: protocol, identities, coordination, and final open evaluation.
- john2: single MLX training trajectory.
- john3: independent maximum-width rank-gradient coverage audit.
- john4: independent bounded score-space convergence audit.

The support audits run concurrently with the one useful training trajectory.
No host duplicates training before the pilot earns confirmation.

## Maximum Compute

One 20-epoch pilot, one open evaluation, two independent launch audits, and
correctness/reporting work. No parameter sweep, duplicate seed, new teacher
compute, sealed test, gameplay, cloud, or external compute.

## Result

Both launch audits passed. On the widest 10,854-action group, all 32 targets
received negative gradients, exactly 32 hard nontargets received positive
gradients, the other 10,758 nontargets received zero gradient, and the update
was finite. Direct bounded optimization again recovered all 12 widest
validation target sets exactly, using at most 9.299 residual points.

No trained epoch beat the untouched warm start. The selected checkpoint
therefore reproduced 29.36% train target recall, 0.18% exact train sets,
26.21% validation target recall, and zero exact validation sets. Four
substantive gates failed.

Rank-boundary loss fell from 1.1915 to 1.0667, but validation recall remained
below baseline throughout and ended at 19.47%. Final winner recall was 55.00%,
confidence coverage 77.92%, and regret 0.163564. Distributing gradient across
the full boundary did not repair neural fit.

The hypothesis is rejected. Uniform set cross entropy, single-extreme
boundary loss, and full rank-matched boundary loss have now all failed while
their score-space targets remain reachable. Another full-network loss swap is
not authorized. The next experiment is a frozen-embedding separability audit:
train only a small probe on cached candidate embeddings to determine whether
the current representation contains the target boundary before changing
architecture or optimizer scope.
