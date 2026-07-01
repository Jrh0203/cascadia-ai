# ADR 0096: Frontier Pre-Pool Candidate Context

Status: complete; rejected as `candidate_projection_insufficient`.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-prepool-context-v1`

## Context

ADR 0094 showed that the selected model's final 192-dimensional output-trunk
embedding cannot fit the open frontier target. ADR 0095 then showed that the
original 148 observable action/prior values, alone or concatenated with that
embedding, also cannot fit it. The best combined probe reached only 30.50%
train target recall and 0.18% exact train sets.

The selected ranker compresses seven candidate-specific streams into a
192-dimensional candidate vector. It then gives every candidate only the
global candidate mean, global candidate maximum, and candidate-minus-mean
before another 768-to-192 output trunk. The unresolved localization is:

1. whether the pre-pool candidate vector already lost the target;
2. whether the final output trunk loses a still-separable signal;
3. whether global mean/max summaries dilute useful candidate-set context; or
4. whether context focused on the observable screen frontier is sufficient.

## Frozen Inputs

- Selected ADR 0089 checkpoint:
  `step-000003592-epoch-0008-batch-000000`.
- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- Reuse the unchanged frontier target, group-balanced binary objective,
  frontier-anchored width-64 selector, checkpoint-selection order, and open
  metrics from ADRs 0094 and 0095.
- No new teacher compute, label, target, split, or selector change.

The sealed test, gameplay, cloud, and external compute remain prohibited.

## Pre-Pool Cache

Refactor `GradedOracleRanker` so the exact 192-dimensional vector immediately
after `candidate_projection` and before candidate pooling can be exported.
The ordinary forward pass must reconstruct its previous outputs bit-for-bit.

Each host independently exports float32 train and validation caches because
all four already hold the immutable checkpoint and datasets. ADR 0094 measured
one local cache extraction at 25.97 seconds, while relaying 2.26 GiB dominated
cluster wall. Local regeneration is therefore the throughput-optimal choice,
not an attempt to create training replicas.

Each cache preserves:

- group offsets;
- frontier target mask;
- source flags;
- observable screen rank;
- selected winner;
- R4800 mean and mask;
- action hash; and
- exact dataset/checkpoint/source identities.

Every open action appears exactly once. Cache features contain no target,
teacher, provenance, selected-winner, or rollout value.

## Frozen Probes

All probes are one-hidden-layer MLX networks with GELU and LayerNorm:
`input -> 256 -> 1`. They use AdamW, learning rate `3e-4`, weight decay
`1e-4`, 20 epochs, and select by train target recall, train exact sets, then
validation target recall.

### Candidate Only

- Host: john1.
- Seed: `2026061613`.
- Input width: 192.
- Input: the pre-pool candidate vector only.
- Question: did `candidate_projection` itself preserve the target?

### Exact Legacy Context

- Host: john2.
- Seed: `2026061614`.
- Input width: 768.
- Input: candidate, global mean, global maximum, candidate minus global mean.
- Question: did the selected 768-to-192 output trunk collapse a separable
  legacy context?

### Rich Moment Context

- Host: john3.
- Seed: `2026061615`.
- Input width: 1,344.
- Input: candidate, mean, maximum, minimum, standard deviation,
  candidate-minus-mean, and candidate-minus-maximum.
- Question: are richer linear-memory global statistics sufficient?

### Screen-Top64 Context

- Host: john4.
- Seed: `2026061616`.
- Input width: 1,344.
- Input: candidate, global mean, global maximum, observable screen-top64 mean,
  observable screen-top64 maximum, candidate-minus-top64-mean, and
  candidate-minus-top64-maximum.
- Screen-top64 membership uses only the frozen observable screen rank and
  stable action-hash tie breaking.
- Question: does relevant-frontier context avoid dilution by thousands of
  clearly weaker legal actions?

## Classification Gates

Every probe must score every action once, produce finite scores, stay below
4 GiB RSS, use zero process swaps, and keep sealed domains unopened.

A probe fits train only if:

- train target-positive recall is at least 80%; and
- train exact target-set recovery is at least 25%.

A fitting probe transfers only if:

- validation target-positive recall is at least 50%; and
- validation exact target-set recovery is at least 1%.

Classify in this order:

1. `candidate_projection_separable`
   - candidate-only train and validation gates pass.
2. `legacy_output_trunk_collapse`
   - candidate-only fails, exact-legacy-context passes.
3. `rich_global_context_sufficient`
   - earlier gates fail, rich-moment context passes.
4. `screen_frontier_context_sufficient`
   - earlier gates fail, screen-top64 context passes.
5. `prepool_train_separable_not_generalized`
   - at least one train gate passes but every transfer gate fails.
6. `candidate_projection_insufficient`
   - every train gate fails.

If multiple mechanisms pass, prefer the earliest classification above; it is
the smallest justified representation change.

## Correctness Gates

- Full Python suite and Ruff pass before real cache generation.
- john1 and john4 independently reconstruct the original residual and
  uncertainty heads from the pre-pool vector on the 10,854-action widest open
  group. Outputs must be bit-identical, finite, below 4 GiB RSS, and swap-free.
- Context builders are permutation equivariant and exactly match their frozen
  widths.
- Padding never affects any pooled statistic.
- Screen-top64 selection is deterministic under rank ties.
- All four source bundles and all portable cache payloads match.

## Cluster Execution

- john1: candidate-only probe and coordination.
- john2: exact-legacy-context probe.
- john3: rich-moment-context probe.
- john4: screen-top64-context probe.
- Each host first creates its own deterministic caches under the host lock and
  `caffeinate`.
- The four distinct probes launch concurrently.
- After training, cross-replay uses a ring:
  john1 on john2, john2 on john3, john3 on john4, john4 on john1.
- A host may begin its available ring replay as soon as both its own probe and
  the incoming artifact are complete.
- Duplicate seeds, same-mechanism replicas, and sweeps remain closed.

Reports record assigned wall time, productive wall time, idle time with work
queued, queue delay, peak RSS, swaps, and scientific hashes per host. The
governing throughput metric is four independently rejected or accepted
representation hypotheses per cluster wall-clock interval.

## Maximum Compute

Eight local cache exports, four one-seed probes, four cross-host replays, two
maximum-width reconstruction audits, tests, and reporting. No extra epoch,
seed, architecture, teacher rollout, sealed test, gameplay, cloud, or
external compute is authorized.

## Consequences

- A passing arm authorizes only the smallest matching trunk treatment.
- Candidate-projection failure authorizes replacing or widening the earlier
  candidate factor integration path.
- No result directly authorizes gameplay or promotion.

## Result

All four frozen probes completed exactly 20 epochs and failed the train-fit
gate:

| Probe | Train recall | Train exact sets | Validation recall | Validation exact sets |
|---|---:|---:|---:|---:|
| candidate only | 28.61% | 0.18% | 24.38% | 0.00% |
| exact legacy context | 29.07% | 0.18% | 24.02% | 0.00% |
| rich moment context | 29.03% | 0.18% | 24.13% | 0.00% |
| screen-top64 context | 28.93% | 0.18% | 24.09% | 0.00% |

Every probe scored all 560 train groups and 240 validation groups exactly
once with finite outputs. Peak process RSS was 2.11 GB, no process swapped,
and the four ring replays reproduced every scientific metric exactly. john1
and john4 also reconstructed the unchanged output trunk and both original
heads bit-for-bit on the 10,854-action maximum-width group.

The preregistered classification is therefore
`candidate_projection_insufficient`. Neither the existing output trunk,
richer global moments, nor observable screen-frontier landmarks recover a
signal absent from the 192-dimensional pre-pool candidate vector.

The probe-and-replay phase completed four independent hypotheses in 1,111.89
seconds, or 12.95 hypotheses per wall-clock hour. No host was idle while
compatible queued work existed. The long john2 idle interval was explicitly
dependency-blocked: its assigned candidate-only replay could not begin until
john1 completed the cluster's slowest training arm.

The next authorized mechanism is a four-way candidate-factor integration
fork at the 1,344-dimensional input to `candidate_projection`. It must compare
distinct pre-compression architectures, use one training treatment per host,
and keep duplicate seeds closed until an arm materially fits train and
transfers to validation.
