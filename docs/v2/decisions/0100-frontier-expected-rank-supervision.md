# ADR 0100: Frontier Expected-Rank Supervision

Status: complete; rejected as `expected_rank_optimization_underfit`.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-expected-rank-v1`

## Context

ADR 0099 rejected the finite-R1200 hard top-64 membership label. Only 10.38%
of validation target slots were statistically separated, 512 teacher
resamples recovered 41.20% of nominal slots, and only 2.50% of complete sets
reproduced. In contrast, uncertainty-aware expected rank preserved 100% of
the open-validation R4800 winner, confidence set, and distinguishable-winner
signal with zero retained regret across every phase.

ADRs 0091-0098 already closed hard-cutoff objectives, boundary losses,
heads, pools, projected factors, raw bypasses, and new constructors. This
pilot therefore changes only the supervision. It does not authorize another
architecture, width, optimizer, or representation experiment.

## Hypothesis

The unchanged public-observable `GradedOracleRanker` can learn a continuous,
uncertainty-aware ordering over the R1200 cohort when statistically
unresolved actions are not forced onto opposite sides of a binary cutoff.
The learned ordering will transfer to unseen open-validation decisions and
retain the R4800 winner inside the exact frontier-anchored width of 64.

## Frozen Inputs

- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Train manifest BLAKE3:
  `7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- Validation manifest BLAKE3:
  `302ceb7a57482b0fb5fb12963521be35aafc121a36f572e6b9f47def1b820a31`.
- The ADR 0081 public feature schema and `GradedOracleRanker` architecture.
- The exact ADR 0089 frontier bit, selector, stable action-hash tie break,
  residual range of plus or minus 12, and proposal width of 64.
- The existing R1200 means, standard deviations, and sample counts.
- The unchanged R4800 evaluation definitions.

The sealed test, gameplay, hidden state, new teacher samples, K2048, cloud,
Modal, and external compute remain prohibited.

## Frozen Expected-Rank Target

For every decision, remove deterministic champion/frontier anchors from the
learned cohort. For every remaining R1200-labeled action, compute:

`expected_rank_i = 1 + sum_j P(value_j > value_i)`

using the exact ADR 0099 independent-normal definition, raw teacher standard
errors, normal-CDF approximation, and stable action hashes. The sum ranges
over R1200-labeled nonfrontier actions only.

Convert the continuous ranks to one normalized target distribution:

`q_i = exp(-(expected_rank_i - 1) / 64) / sum_j exp(-(expected_rank_j - 1) / 64)`

R1200-unlabeled nonfrontier actions receive zero target mass but remain in the
student denominator. The width-scaled exponential is frozen because width 64
is the deployed decision budget: it supplies smooth pressure on the complete
teacher cohort while concentrating most mass near the retained boundary.

The student distribution is the temperature-2 softmax of the unchanged
`screen_value + bounded_residual` score over every nonfrontier action. The
only training loss is cross entropy from `q` to that student distribution.
No hard target membership, R4800 label, auxiliary regression, pair mining,
screen-only regularizer, confidence weighting, or target-temperature sweep
is allowed.

## Target Cache Contract

Expected ranks are computed once per immutable open dataset and stored in a
small sidecar cache. The cache contains group IDs, candidate counts, offsets,
and one float32 expected rank per aligned candidate, with noneligible rows
represented explicitly as nonfinite.

The cache manifest binds:

- experiment and schema version;
- source dataset ID, split, manifest digest, group count, candidate count,
  and shard identities;
- expected-rank formula and target temperature 64;
- ordered group IDs, candidate counts, action hashes, masks, and rank bytes;
- generator source identity and complete-file SHA256 and BLAKE3 digests.

john2 builds the trainer cache. john1 independently rebuilds both open caches
from the immutable datasets. Training cannot start until the two scientific
payloads and rank-array bytes are identical. Caches are an execution
optimization, not a new label source.

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

The selected checkpoint is the one evaluated for all gates. No warm start,
second seed, architecture change, head change, loss mixture, learning-rate
sweep, or patience change is authorized.

## Independent Diagnostics

### Cache Reproduction

- Host: john1.
- Independently rebuild train and validation target caches with eight ordered
  CPU workers.
- Require byte-identical target arrays and identical scientific manifests to
  john2 before training.

### Optimization And Gradient Audit

- Host: john3.
- Use the widest train group and the frozen seed/model/loss.
- Report initial loss, finite whole-model and residual-head gradient norms,
  residual distribution, and expected-rank target recovery.
- Run exactly 32 AdamW steps on that one group at the frozen optimizer
  settings and report final loss and target recovery.
- Pass only if gradients are finite and nonzero, loss decreases strictly,
  peak RSS is at most 4 GiB, and process and system swap do not increase.
- This is a bounded mechanistic audit, not a second training replica.

### Baseline And Error Anatomy

- Host: john4.
- Evaluate the unchanged anchored screen against the expected-rank target on
  every train and validation group.
- Report target-slot recall, exact target sets, rank correlation, target
  entropy, frontier counts, R4800 retention metrics, and phase, Nature Token,
  draft-family, and group-width slices.
- This audit uses no learned treatment weights and cannot select or alter the
  target.

All three diagnostics launch as soon as their immutable inputs are available
and run concurrently with compatible work. Duplicate model training is
prohibited.

## Pilot Gates

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
- the john3 optimization audit passes;
- selected-checkpoint metrics reproduce bit-identically on john1;
- at least 20,000 action scores per second, P99 decision latency at most
  250 ms, peak RSS at most 4 GiB, zero process swaps, and no positive system
  swap delta on john1 and john2; and
- sealed test, gameplay, new teacher compute, and external compute remain
  unopened.

Passing authorizes only a separately preregistered sealed-test evaluation.
It does not authorize gameplay automatically.

## Classification

1. `expected_rank_model_sufficient` if every pilot gate passes.
2. `expected_rank_train_fit_only` if both train-fit gates pass but any
   validation quality gate fails.
3. `expected_rank_optimization_underfit` if either train-fit gate fails while
   cache integrity and the optimization audit pass.
4. `expected_rank_pipeline_invalid` if cache identity, finite gradients,
   one-step semantics, source identity, or complete-coverage integrity fails.

The classification is mechanical. Thresholds and precedence may not change
after treatment metrics are visible.

## Pre-Result Cache-Lookup Correction

The first training and baseline launch failed before producing any treatment
metric. Dataset group IDs are unsigned 64-bit values in the binary format, but
the decoded MLX batch stores their bit pattern in signed `int64`. Cache keys
were correctly written as unsigned values, while lookup converted negative
decoded values directly to Python integers.

Before any model validation or baseline result was written, lookup was
corrected to normalize the signed bit pattern modulo `2^64`. A regression test
now covers a high-bit group ID. The cache bytes, expected-rank formula,
treatment, seed, optimizer, thresholds, and all scientific definitions remain
unchanged.

The entire first wave, including the successful low-ID widest-group diagnostic,
was quarantined under `invalid-launch-group-id-sign/` so the corrected source
bundle is common to every accepted job. Corrected source identity and all
focused tests must pass on all four Macs before relaunch.

## Cluster Allocation

| Host | Primary work | Follow-on work |
|---|---|---|
| john1 | independent target-cache reproduction | selected-checkpoint replay and performance |
| john2 | canonical target cache, then one MLX pilot | origin evaluation and performance |
| john3 | maximum-width optimization/gradient audit | report verification |
| john4 | baseline/generalization error anatomy | report verification |

Use one MLX process on john2. CPU diagnostics use eight ordered workers where
parallel group work exists, leaving two physical cores for orchestration.
Every remote long-running command uses a host lock and `caffeinate`.

## Stop Rule

Run the frozen cache pair, one john3 optimization audit, one john4 baseline
audit, and one john2 training seed. Stop training after 20 epochs or six
consecutive non-improving validation epochs. If the pilot fails, do not run a
second seed, warm start, architecture treatment, or sealed evaluation.

## Maximum Compute

Two independently generated cache pairs, one 32-step maximum-width audit, one
complete baseline audit, one at-most-20-epoch MLX pilot, one selected
checkpoint replay, two performance measurements, correctness tests, and one
combined report. No new teacher data, duplicate training, sealed test,
gameplay, K2048, cloud, Modal, or external compute is authorized.

## Result

The corrected experiment completed all authorized work. john1 and john2
independently generated byte-identical train and validation target caches.
john3's widest-group gradient audit passed with finite nonzero whole-model and
residual-head gradients, reducing loss from 7.761970 to 5.597785 in 32 steps.
john4's complete baseline audit covered every open group and candidate without
swap.

The sole john2 trainer completed the full 20-epoch budget. Epoch 15 was selected
at checkpoint `step-000006755-epoch-0015-batch-000000`. The selected model
replayed bit-identically on john1:

- train expected-rank target recall: 32.21% versus the 80% gate;
- train exact target sets: 0.18% versus the 25% gate;
- validation expected-rank target recall: 27.81% versus the 50% gate;
- validation exact target sets: 0.42% versus the 1% gate;
- validation R4800 winner recall: 73.33%;
- validation confidence-set coverage: 88.33%;
- validation distinguishable-winner recall: 86.84%; and
- validation retained regret: 0.070215.

Both origin and replay passed inference performance, integrity, finite-score,
memory, and process-swap gates. The sealed test, gameplay, new teacher compute,
cloud, and external compute remained closed. The mechanical classification is
`expected_rank_optimization_underfit`.

Exploratory post-launch diagnostics explain the miss without changing the
classification. The frozen scale-64 target places only 44.84% of train and
45.51% of validation probability mass inside the deployed nonfrontier target
set. At a uniform student, only 25.64% and 26.12% of absolute gradient falls
inside that set. The bounded score space is not the constraint: a symmetric
residual range of 6 recovers every train and validation target set.

A deterministic concentration audit found that expected-rank scale 16 places
93.76% of train and 93.75% of validation target mass inside the deployed set,
while preserving the same continuous expected-rank ordering and width-64
ceiling. This authorizes one separately preregistered scale-16 pilot. It does
not authorize an optimizer, architecture, width, seed, or broad temperature
sweep.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-expected-rank-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-expected-rank-v1-result.md`.
