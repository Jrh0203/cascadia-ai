# ADR 0097: Frontier Candidate-Factor Integration

Status: complete; rejected as `candidate_factor_inputs_insufficient`.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-factor-integration-v1`

## Context

ADR 0096 exported the exact 192-dimensional vector produced by
`candidate_projection` and tested candidate-only, legacy mean/max, richer
global moments, and observable screen-top64 context. Every arm failed to fit
the open train target at 28.61%-29.07% recall and 0.18% exact sets. The
preregistered classification was `candidate_projection_insufficient`.

The selected model builds that vector by concatenating seven 192-dimensional
candidate factors and compressing 1,344 values through
`1344 -> 576 -> 192`:

1. action;
2. prior;
3. parent state;
4. staged post-draft state;
5. action-to-board cross attention;
6. action-to-staged-market cross attention; and
7. action times parent.

The next localization asks whether those exact pre-compression factors
preserve the frontier target and which linear-memory integration mechanism,
if any, can recover it.

## Frozen Inputs

- Selected ADR 0089 checkpoint:
  `step-000003592-epoch-0008-batch-000000`.
- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- Unchanged frontier target, group-balanced binary objective,
  frontier-anchored width-64 selector, selection order, and open metrics from
  ADRs 0094-0096.
- No new teacher compute, labels, target, split, or selector.

The sealed test, gameplay, cloud, and external compute remain prohibited.

## Factor Cache

Refactor `GradedOracleRanker` to export the exact seven factor tensors before
their concatenation and before `candidate_projection`. The ordinary forward
pass must reconstruct its previous pre-pool vector, output embedding,
residual, and standard-error outputs bit-for-bit.

Each host independently writes a manifest-backed float32 cache with shape
`candidate_count x 7 x 192`, flattened only for storage. The expected train
plus validation feature payload is approximately 15.0 GiB per host. Local
regeneration is preferred over relaying that payload.

The cache preserves the same group offsets, frontier target, source flags,
observable screen rank, selected winner, R4800 values, and action hash as ADR
0096. Factor features contain no target, teacher value, selected winner,
provenance flag, or rollout result.

## Frozen Probes

All probes train for exactly 20 epochs with AdamW, learning rate `3e-4`,
weight decay `1e-4`, and the unchanged group-balanced binary loss. Selection
uses train target recall, train exact sets, then validation target recall.

### Wide Concatenation

- Host: john1.
- Seed: `2026061617`.
- Flatten the seven factors to 1,344 values.
- Architecture:
  `1344 -> 1024 -> 512 -> 1`, with GELU and LayerNorm between linear layers.
- Question: is the target directly separable before the 192-dimensional
  bottleneck?

### Screen-Relative Factor Context

- Host: john2.
- Seed: `2026061618`.
- Reduce each candidate's flattened factors to 384 values.
- Build candidate, global mean, global maximum, observable screen-top64 mean,
  observable screen-top64 maximum, candidate-minus-top64-mean, and
  candidate-minus-top64-maximum.
- Architecture:
  `2688 -> 768 -> 384 -> 1`, with GELU and LayerNorm.
- Screen landmarks use only frozen observable rank and stable action-hash
  tie breaking.
- Question: does set context work when applied before lossy compression?

### Factor-Token Attention

- Host: john3.
- Seed: `2026061619`.
- Treat the seven 192-dimensional factors as typed tokens.
- Add learned factor-type embeddings and apply two six-head pre-norm
  self-attention blocks with feed-forward multiplier four.
- Concatenate token mean and maximum, then use `384 -> 256 -> 1`.
- Question: are learned factor-to-factor interactions the missing mechanism?

### Pairwise-Gated Factors

- Host: john4.
- Seed: `2026061620`.
- Apply one identity-specific `192 -> 256` projection to each factor.
- Combine a learned softmax-weighted factor sum, the normalized sum of all 21
  elementwise pair products, and factorwise maximum.
- Architecture:
  `768 -> 512 -> 256 -> 1`, with GELU and LayerNorm.
- Question: do explicit pairwise interactions recover signal that dense
  compression loses?

## Classification Gates

Every probe must score every action once per evaluation, produce finite
scores, stay below 6 GiB peak process RSS, use zero process swaps, and keep
sealed domains unopened.

A probe fits train only if:

- train target-positive recall is at least 80%; and
- train exact target-set recovery is at least 25%.

A fitting probe transfers only if:

- validation target-positive recall is at least 50%; and
- validation exact target-set recovery is at least 1%.

Classify in deployment-complexity order:

1. `wide_concat_sufficient`;
2. `pairwise_factor_sufficient`;
3. `factor_attention_sufficient`;
4. `screen_relative_factor_context_sufficient`;
5. `candidate_factors_train_separable_not_generalized`; or
6. `candidate_factor_inputs_insufficient`.

If multiple mechanisms pass, select the earliest classification. A pass
authorizes only the matching end-to-end candidate integration treatment.

## Correctness Gates

- Full Python suite and Ruff pass before real factor-cache generation.
- The refactored ordinary forward pass is bit-identical on unit fixtures.
- john1 and john4 independently reconstruct the original
  `candidate_projection`, output trunk, residual head, and standard-error head
  on the 10,854-action maximum-width group.
- Every architecture is candidate-permutation equivariant; the
  screen-relative arm is group-permutation equivariant and deterministic
  under rank ties.
- Padding never contributes to factor or group pooling.
- All four source bundles and portable cache payloads match.
- Each host has at least 24 GiB free before cache generation and remains
  swap-free.

## Cluster Execution

- john1: wide concatenation plus coordination.
- john2: screen-relative context, the expected heaviest MLX arm.
- john3: factor-token attention.
- john4: pairwise-gated factors.
- All hosts generate local caches concurrently under host locks and
  `caffeinate`.
- All four distinct probes then launch concurrently.
- Ring cross-replay is john1 to john2, john2 to john3, john3 to john4, and
  john4 to john1.
- A host begins available replay as soon as its own probe and incoming
  artifact are complete.
- Duplicate seeds, same-mechanism replicas, and sweeps remain closed.

Reports record assigned wall time, productive wall time, dependency-blocked
idle, idle with compatible work queued, peak RSS, swaps, cache throughput,
candidate throughput, and scientific hashes. The governing throughput metric
is independently resolved architecture hypotheses per wall-clock hour.

## Stop Rule

Run exactly 20 epochs per arm. Do not resize, continue, reseed, or alter an
architecture after metrics are visible. Classify only after all four reports
and ring replays pass integrity checks.

## Maximum Compute

Eight local factor-cache exports, four one-seed probes, four cross-host
replays, two maximum-width reconstruction audits, tests, and reporting. No
extra epoch, seed, architecture, teacher rollout, sealed test, gameplay,
cloud, or external compute is authorized.

## Consequences

- A passing arm authorizes the smallest matching end-to-end integration
  treatment.
- If every train gate fails, the seven upstream factor representations are
  insufficient and the next experiment must move into factor construction,
  not add another integration head.
- No result directly authorizes gameplay or promotion.

## Result

All four probes completed exactly 20 epochs and failed the train-fit gate:

| Probe | Train recall | Train exact sets | Validation recall | Validation exact sets |
|---|---:|---:|---:|---:|
| wide concatenation | 30.00% | 0.00% | 25.07% | 0.00% |
| screen-relative factor context | 30.88% | 0.18% | 25.45% | 0.00% |
| factor-token attention | 29.39% | 0.18% | 24.99% | 0.00% |
| pairwise-gated factors | 30.66% | 0.18% | 24.68% | 0.00% |

Every probe scored all 560 train groups and 240 validation groups exactly
once with finite outputs. The four ring replays reproduced every scientific
metric exactly. Peak active MLX memory was 3.92 GB during training, all phase
boundaries retained zero cached bytes, and no process swapped.

The first operational launch was invalidated after MLX's default reusable
buffer cache caused john1 to retain approximately 11.3 GiB and enter severe
swap. That launch completed no arm and contributed no selection data. The
permanent correction capped free-buffer caching at 512 MiB, added allocator
telemetry and maximum-width backward audits, and preserved loss and gradients
bit-for-bit. The corrected john1 arm ran more than ten times faster.

The preregistered classification is therefore
`candidate_factor_inputs_insufficient`. The 1,344-dimensional concatenation
of action, prior, parent, staged state, board cross-attention, staged-market
cross-attention, and action-parent product does not preserve enough
information to recover the frontier target. Dense width, candidate-relative
context, factor self-attention, and explicit pairwise interactions are closed.

Probe plus replay wall time was 3,944.72 seconds, resolving 3.65 independent
hypotheses per cluster wall-clock hour. john1 and john2 began available ring
replays while john3 continued the critical attention arm; the final two
replays were pre-positioned and launched immediately when dependencies
cleared.

The next authorized experiment must move upstream into factor construction.
It should test raw action and staged-state observables plus newly learned
candidate-to-board and candidate-to-market relations before the current
192-dimensional factor projections. Adding another head, pool, or integration
mechanism over the frozen seven factors is prohibited.
