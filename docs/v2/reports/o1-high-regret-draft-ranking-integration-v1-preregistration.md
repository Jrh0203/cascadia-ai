# O1 High-Regret Draft-Ranking Integration v1 Preregistration

Experiment: `o1-high-regret-draft-ranking-integration-v1`

ADR: `0188-o1-high-regret-draft-ranking-integration.md`

Status: completed with `o1_ranking_validation_factorial_null`; sealed test not
opened

## Hypothesis

Opponent-sensitive future-access probabilities contain action-ranking
information absent from the accepted exact-R2 candidate representation.
Candidate-aligned A2 probabilities should reduce retained oracle regret more
than an adapter-only control, A0 public-state probabilities, or an
alignment-destroying A2 shuffle.

## Frozen Upstream Evidence

O1 terminal classification:

`opponent_intent_policy_holdout_replication_passed`

Classification ID:

`5676f5025317bcd1543bf1d8b8b6f80df3dae09714dba4f945b6a4283aae24aa`

Selected O1 model:

`artifacts/experiments/o1-opponent-intent-mlx-factorial-v1/collected/a2-primary/training/final-model.safetensors`

O1 control model:

`artifacts/experiments/o1-opponent-intent-mlx-factorial-v1/collected/a0-primary/training/final-model.safetensors`

Frozen exact-R2 warm start:

`artifacts/experiments/r3-action-edit-mlx-comparison-v1/runs/c0_full_r2_afterstate/checkpoints/step-000003000-epoch-0000-batch-003000/model.safetensors`

Warm-start report ID:

`75f2daf1ed8c70ac6fdeecb43f2a54efb0850c0969bc83a7f1ae0e08a569f562`

## Frozen Data

Open train:

- seven games;
- 560 decisions;
- 2,135,111 complete legal actions;
- accepted R3 cache retains 280,012 actions before top-64 selection.

Open validation:

- three games;
- 240 decisions;
- 860,203 complete legal actions.

Sealed test:

- three games;
- 240 decisions;
- 876,765 complete legal actions;
- no model score, cohort, feature, metric, or checkpoint may touch it before
  validation authorization.

All games are four-player AAAAA with habitat bonuses disabled.

## Frozen Cohort Selection

The exact-R2 warm start scores candidates in canonical orientation.

- Train uses top 63 plus the R4800 selected candidate if needed.
- Validation uses strict top 64.
- Test, if authorized, uses strict top 64.
- Ties use canonical action hash.
- Cohort width is exactly 64 for every decision.

The selector writes immutable source indices, retained-cache positions, base
scores, ranks, action hashes, group IDs, game IDs, turns, and cohort hashes.
Train and validation selection must reproduce bit-for-bit on a second host.

## Frozen Refill Marginalization

Eight proposal samples are generated for every candidate. The seed domain is:

`cascadia-v2-o1-ranking-public-refill-v1`

Each sample key includes split, group ID, action hash, and proposal index.
Tile archetypes are sampled from the S1 exact public archetype distribution.
Wildlife are sampled from staged public bag counts. Sampling is without
replacement when more than one component of the same kind is missing.

The sampler may fill only components removed by the candidate action. It may
not modify placed boards, Nature Tokens, turn metadata, surviving market
components, or public action history.

## Frozen Intent Vector

Each O1 model runs on every completed proposal. Softmax probabilities are
averaged across proposals, then flattened in this exact order:

1. disposition `[4 slots, 4 classes]`;
2. pair-survival positive probability `[4]`;
3. final-slot `[4 slots, 4 classes]`;
4. next-opponent tile-slot `[3, 4]`;
5. next-opponent wildlife-slot `[3, 4]`;
6. next-opponent independent-draft probability `[3]`;
7. next-opponent drafted-wildlife `[3, 5]`;
8. next-opponent free-replacement probability `[3]`.

Total width: 81 float32 values.

S3 uses a deterministic derangement within:

- split;
- phase quartile;
- paired versus independent candidate draft;
- paired-depletion versus independent-depletion market shape.

No stratum may retain a fixed point. Singleton strata are merged with the
adjacent phase stratum before permutation.

## Frozen Model

Architecture:

`o1-intent-conditioned-exact-r2-reranker-v1`

All arms load the same exact-R2 warm start. Existing modules and output heads
are frozen. Trainable modules are:

- `intent_projection`: 81 -> 128 -> 64;
- `intent_fusion`: concatenated base, intent, and interaction
  192 -> 128 -> 64;
- `intent_delta`: 64 -> 64, zero initialized.

The contextualized candidate encoding is:

`base_hidden + intent_delta(intent_fusion(...))`

Z0 supplies zeros. B1, P2, and S3 differ only in which immutable 81-value
tensor is routed into the same graph.

## Frozen Optimization

- seed: `2026061719`;
- optimizer: AdamW;
- steps: 2,000;
- groups per step: 4;
- candidates per group: 64;
- learning rate: `1e-4`;
- weight decay: `1e-4`;
- checkpoint interval: 250;
- metric interval: 100;
- geometry transform: canonical only;
- base and output heads frozen;
- adapter only trainable;
- no validation during training;
- no early stopping;
- final step only.

## Frozen Metrics

Primary:

- mean top-1 retained R4800 regret.

Paired inference:

- treatment minus Z0 regret by group;
- game-clustered bootstrap;
- 20,000 replicates;
- seed `2026061720`.

Secondary:

- top-1 R4800-winner recall;
- R1200 pairwise ordering accuracy;
- mean regret among groups with Z0 regret at least 0.50;
- opening, early-middle, late-middle, and endgame;
- low supply;
- Nature Token available;
- independent-draft winner;
- candidates per second, decision latency, peak active MLX memory, RSS, and
  swap.

## Frozen Validation Classification

Every primary and rotated-host report, final model, trainable tensor, cohort,
intent cache, and prediction panel must match.

Aligned-arm eligibility and selector order are exactly ADR 0188.

Terminal validation classifications:

- `o1_ranking_validation_arm_selected`;
- `o1_ranking_validation_factorial_null`;
- `o1_ranking_validation_invalid`.

## Frozen Test Classification

The test split is opened only after
`o1_ranking_validation_arm_selected`.

Terminal classifications:

- `o1_ranking_policy_holdout_transfer_passed`;
- `o1_ranking_policy_holdout_transfer_failed`;
- `o1_ranking_test_not_opened`;
- `o1_ranking_test_invalid`.

## Claim Boundary

A replicated offline pass authorizes only a separately preregistered bounded
gameplay experiment. It does not authorize direct deployment, champion
promotion, or a claim that mean score improved.

## Terminal Result

All primary and rotated-host artifacts matched exactly. P2 was directionally
best but improved mean retained R4800 regret by only `0.009142`, with paired
95% interval `[-0.018194, 0.000000]` and high-regret improvement `0.022008`.
No treatment met the frozen eligibility gates.

The validation classifier returned `o1_ranking_validation_factorial_null`.
The test classifier returned `o1_ranking_test_not_opened`. See
`o1-high-regret-draft-ranking-integration-v1-result.md`.
