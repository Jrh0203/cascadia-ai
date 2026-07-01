# ADR 0101: Frontier Expected-Rank Scale 16

Status: complete; rejected as `scale16_alignment_insufficient`.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-expected-rank-scale16-v1`

## Context

ADR 0100 tested continuous uncertainty-aware expected-rank supervision with
target scale 64. The selected model reached 32.21% train target recall and
27.81% validation target recall, with 0.18% and 0.42% exact target sets. The
pipeline, cache, gradient, replay, inference, memory, and finite-score gates
passed, so the mechanical classification was
`expected_rank_optimization_underfit`.

The post-launch mechanism audit found a specific mismatch. Scale 64 places
only 44.84% of train and 45.51% of validation target probability mass inside
the exact nonfrontier set retained by the deployed width-64 selector. At a
uniform student, only 25.64% and 26.12% of absolute gradient acts inside that
set. Scale 16 raises deployed-set target mass to 93.76% on train and 93.75%
on validation while preserving the same expected-rank order. A symmetric
residual range of 6 can recover every open target set, so score range is not
the active constraint.

This experiment tests that single alignment correction. It is not an
architecture, optimizer, width, seed, representation, or broad temperature
sweep.

## Hypothesis

Concentrating the continuous expected-rank target at scale 16 will align most
cross-entropy pressure with the exact width-64 deployment decision. The
unchanged public-observable `GradedOracleRanker` will materially fit the open
train target and transfer to open validation while retaining the R4800 winner.

## Frozen Inputs

- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Train manifest BLAKE3:
  `7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- Validation manifest BLAKE3:
  `302ceb7a57482b0fb5fb12963521be35aafc121a36f572e6b9f47def1b820a31`.
- The ADR 0081 public feature schema and unchanged
  `GradedOracleRanker` architecture.
- The exact ADR 0089 frontier bit, selector, stable action-hash tie break,
  residual range of plus or minus 12, and proposal width of 64.
- The ADR 0099 independent-normal expected-rank formula and the existing
  R1200 means, standard deviations, and sample counts.
- The unchanged R4800 evaluation definitions.

The sealed test, gameplay, hidden state, new teacher samples, K2048, cloud,
Modal, and external compute remain prohibited.

## Single Treatment Change

For every R1200-labeled nonfrontier action, retain the exact ADR 0100 expected
rank:

`expected_rank_i = 1 + sum_j P(value_j > value_i)`

Change only the normalized target distribution:

`q_i = exp(-(expected_rank_i - 1) / 16) / sum_j exp(-(expected_rank_j - 1) / 16)`

R1200-unlabeled nonfrontier actions receive zero target mass but remain in the
student denominator. The student remains the temperature-2 softmax of
`screen_value + bounded_residual` over every nonfrontier action. The only loss
is cross entropy from `q` to that student distribution.

No hard membership target, R4800 label, auxiliary loss, pair mining, screen
regularizer, confidence weighting, target-scale sweep, student-temperature
change, or curriculum is allowed.

## Cache Contract

Build new experiment-scoped train and validation cache manifests with target
scale 16. The expected-rank arrays must be byte-identical to ADR 0100 because
the rank formula and source data are unchanged. The manifests must bind the
new experiment ID, target scale, dataset identity, ordered group and action
identity, complete file digests, and source identity.

john2 builds the canonical cache pair. john1 independently rebuilds both
caches. Training cannot begin until:

- john1 and john2 scale-16 rank arrays are byte-identical;
- both pairs cover every open group and candidate;
- both rank arrays are byte-identical to the corresponding ADR 0100 arrays;
  and
- the only scientific manifest differences are experiment identity and target
  scale.

## Frozen Model And Optimization

- Host: john2.
- Seed: `2026061626`.
- Initialize the unchanged `GradedOracleRanker` from scratch.
- AdamW learning rate `1e-4`, weight decay `1e-4`.
- At most 20 epochs with six-epoch validation patience.
- Group batch size 64, packed action limit 8,192, hard group ceiling 16,384.
- Exact uniform hex-rotation augmentation.
- Checkpoint every 250 optimizer steps and every completed epoch.
- Select lowest validation expected-rank target-slot miss rate, then higher
  exact expected-rank target-set recovery, then lower retained R4800 regret.

No warm start, second seed, architecture change, head change, optimizer
change, loss mixture, or patience change is authorized.

## Independent Diagnostics

### Target And Cache Alignment

- Host: john1.
- Independently rebuild both scale-16 caches.
- Measure target mass, entropy, effective support, and uniform-student
  gradient allocation on every open group.
- Require mean deployed-set target mass above 90% on train and validation.
- Compare the rank arrays byte-for-byte with ADR 0100.

### Multi-Group Optimization Audit

- Host: john3.
- Use the frozen seed, model, scale-16 loss, and optimizer.
- Audit the 12 widest train groups independently for 32 AdamW steps each.
- Report initial and final loss, finite whole-model and residual-head gradient
  norms, target recall, exact sets, peak RSS, and swap.
- Pass only if all 12 groups have finite nonzero gradients, every final loss
  is strictly lower than its initial loss, peak RSS is at most 4 GiB, and
  process and system swap do not increase.

### Baseline And Reachability Anatomy

- Host: john4.
- Evaluate the unchanged anchored screen against the scale-16 objective and
  expected-rank deployment target on every train and validation group.
- Report objective, target recall, exact sets, rank correlation, R4800
  retention, phase, Nature Token, draft-family, and group-width slices.
- Reconfirm exact target-set reachability at residual ranges 0, 3, 6, and 12.

These diagnostics are distinct scientific jobs, not replicas of the trainer.
Once a diagnostic finishes, its host immediately advances report
verification, selected-checkpoint replay preparation, or the next
launch-ready dependency. Duplicate model training is prohibited.

## Promotion Gates

The selected checkpoint advances only if all integrity and cache gates pass
and complete open-split evaluation achieves:

- train expected-rank target-slot recall at least 80%;
- train exact expected-rank target-set recovery at least 25%;
- validation expected-rank target-slot recall at least 50%;
- validation exact expected-rank target-set recovery at least 1%;
- validation R4800-winner recall strictly greater than 98%;
- validation 95% confidence-set coverage at least 99%;
- validation distinguishable-winner recall at least 98%;
- validation retained mean R4800 regret below 0.03;
- early, middle, and late winner recall and confidence coverage each at least
  98%, with retained regret below 0.03;
- every eligible Nature Token and independent-draft subset with at least 20
  groups reaches 95% winner recall and retained regret below 0.25;
- every group and candidate is scored exactly once with finite scores;
- all independent diagnostics pass;
- selected-checkpoint metrics reproduce bit-identically on john1;
- at least 20,000 action scores per second, P99 decision latency at most
  250 ms, peak RSS at most 4 GiB, zero process swaps, and no positive system
  swap delta on john1 and john2; and
- sealed test, gameplay, new teacher compute, and external compute remain
  unopened.

Passing authorizes only a separately preregistered sealed-test evaluation.

## Classification

1. `scale16_expected_rank_model_sufficient` if every promotion gate passes.
2. `scale16_expected_rank_train_fit_only` if both train-fit gates pass but any
   validation quality gate fails.
3. `scale16_alignment_material_but_underfit` if cache and optimization
   integrity pass, either train-fit gate fails, and selected train target
   recall is at least 42.21%, ten percentage points above ADR 0100.
4. `scale16_alignment_insufficient` if cache and optimization integrity pass,
   either train-fit gate fails, and selected train target recall is below
   42.21%.
5. `scale16_pipeline_invalid` if cache identity, source identity, finite
   gradients, complete coverage, replay, or boundary integrity fails.

The thresholds and precedence are mechanical and may not change after
treatment metrics are visible.

## Cluster Allocation

| Host | Primary work | Follow-on work |
|---|---|---|
| john1 | independent caches and split-wide alignment audit | selected-checkpoint replay |
| john2 | canonical caches and sole MLX trainer | origin evaluation |
| john3 | 12-group optimization audit | trajectory and report verification |
| john4 | baseline and reachability anatomy | subset and report verification |

Use one MLX process on john2. CPU jobs use measured worker counts with bounded
memory and no swap. Every remote long-running command uses the host lock and
`caffeinate`. No healthy host waits behind the trainer while compatible
declared work is queued.

## Stop Rule

Run one frozen seed. Stop after 20 epochs or six consecutive non-improving
validation epochs. If the pilot fails, do not run a second seed, warm start,
scale variant, architecture treatment, optimizer treatment, or sealed
evaluation. Classify only after origin evaluation and john1 replay complete.

## Maximum Compute

Two independently generated cache pairs, one split-wide alignment audit, one
12-group 32-step optimization audit, one complete baseline and reachability
audit, one at-most-20-epoch MLX pilot, one selected-checkpoint replay, two
performance measurements, correctness tests, source identity checks, and one
combined report. No new teacher data, duplicate training, sealed test,
gameplay, K2048, cloud, Modal, or external compute is authorized.

## Result

The experiment completed all authorized work. john1 and john2 independently
generated byte-identical train and validation rank arrays, and those arrays
were byte-identical to ADR 0100. The only target change was the preregistered
scale from 64 to 16. Source identity matched across all four Macs.

The scale-16 alignment audit passed:

- train deployed-set target mass: 93.76%;
- validation deployed-set target mass: 93.75%;
- train uniform-start absolute gradient inside the deployed set: 47.90%; and
- validation uniform-start absolute gradient inside the deployed set: 47.86%.

john3's 12 widest-group audit passed every gradient, loss, memory, and swap
gate. Each group reduced loss strictly over 32 steps. john4's complete
baseline and reachability audit scored every open group and candidate once,
and a symmetric residual range of 6 recovered every train and validation
target set.

The sole john2 trainer stopped after epoch 16 under the frozen six-epoch
futility rule. Epoch 10 was selected at checkpoint
`step-000004514-epoch-0010-batch-000000`. Its origin evaluation and john1
replay were scientifically bit-identical:

- train expected-rank target recall: 30.23% versus the 80% gate;
- train exact target sets: 0.18% versus the 25% gate;
- validation expected-rank target recall: 27.21% versus the 50% gate;
- validation exact target sets: 0%;
- validation R4800 winner recall: 73.33%;
- validation confidence-set coverage: 89.58%;
- validation distinguishable-winner recall: 89.47%; and
- validation retained regret: 0.071272.

Selected train recall was 1.98 percentage points below ADR 0100 and 11.98
points below the 42.21% material-alignment threshold. Scale 16 therefore
corrected target concentration but did not improve full-dataset fit. The
mechanical classification is `scale16_alignment_insufficient`.

The campaign completed in 2,292.16 seconds with one trainer, zero duplicate
training, no new teacher compute, and no sealed-test, gameplay, cloud, Modal,
or external compute use.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-expected-rank-scale16-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-expected-rank-scale16-v1-result.md`.
