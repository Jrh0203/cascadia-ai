# ADR 0081: Complete-Action Graded Oracle Ranker

Status: closed rejected on validation; sealed test and gameplay unopened.

Date: 2026-06-15

Pretraining corrections: ADR 0084 supersedes only the collection-index
identity and complete-group packing limit. ADR 0085 makes the intended
observable-only model boundary explicit after an invalid implementation
diagnostic exposed teacher-provenance leakage. Neither correction changes the
source records, split, labels, loss, optimizer, seeds, or gates.

Experiment ID: `complete-action-graded-oracle-ranker-v1`

## Context

The complete-action K1024 public oracle recalled 99.375% of its online
high-confidence winners, but its 12-game pilot stopped at 98.417 mean and
+2.854 paired points. Increasing screen width again is not authorized. The
remaining question is whether a learned scorer can use the full shape of the
graded R1200/R4800 evidence instead of copying a single noisy winner or
ranking actions only by the historical screen.

Earlier full-legal apprentices are not an adequate test of this mechanism.
Winner-only imitation discarded value margins and uncertainty. ADR 0053
retained only a sampled 96-action set and used K32/R600 evidence. This
experiment instead preserves every canonical action in each K1024 audit
decision, every dense screen estimate, every available R600/R1200/R4800
estimate, exact action deltas, and the observable state needed to interpret
the action.

## Source Corpus

The immutable source is:

`artifacts/datasets/full-legal-screen-width-recovery-v1`

- Source experiment:
  `full-legal-screen-width-recovery-v1-20260615`.
- Rules: four-player AAAAA, habitat bonuses disabled.
- Games: raw numeric seeds `61000-61012`.
- Decisions: 1,040.
- Canonical actions: 3,872,079.
- Screen width used for R1200 allocation: 1,024.
- R1200 budget: 1,200 full-terminal samples.
- R4800 budget: 4,800 full-terminal samples.
- Source manifest SHA-256:
  `c5e568644e1f7d2d11eed7b6099778853ecd4898cd586f31b5edaddd566fdca5`.
- Source manifest BLAKE3:
  `2751d33ae1da1d9cad4355555c29f988d69caf93c96158fd8d6f7d6499415ca7`.
- Collection index SHA-256:
  `c2385cf34508834b14b24a860d37b48cb94d3456918cf8bf28be4c038ba46610`.
- Collection index BLAKE3:
  `e6a369788acd81e560e7f2599c621e387ca6d0e4944fe610438d532ae3bbe59d`.

The game-disjoint split is frozen:

| Split | Seeds | Games | Host balance |
|---|---|---:|---|
| Train | `61000-61002,61005-61006,61009-61010` | 7 | 3/2/2 |
| Validation | `61003,61007,61011` | 3 | 1/1/1 |
| Sealed test | `61004,61008,61012` | 3 | 1/1/1 |

No decision from one game may appear in another split. Test conversion is
allowed before training only as a checksummed opaque artifact. No model,
initialization, threshold, feature decision, or checkpoint may be evaluated
against test records until ADR 0082 authorizes access.

## Lossless Dataset Contract

Add a versioned grouped binary format. Each group stores one observable parent
position and every canonical action in the source decision. It must preserve:

- raw seed, completed turn, acting seat, phase, group ID, and source hashes;
- the complete shared public `PositionRecord`;
- the complete public supply vector;
- the exact ordered `TurnAction`, including free refresh and every paid-wipe
  mask, with capacity for all 20 Nature Tokens;
- the staged observable market and public supply after the action prelude;
- draft slots, tile identity and terrain metadata, wildlife identity, tile
  coordinate, rotation, wildlife coordinate, and action family;
- exact habitat, Bear, Elk, Salmon, Hawk, Fox, Nature Token, and total score
  deltas;
- historical model immediate value, remaining value, total screen value, and
  complete-screen rank;
- visible wildlife count, public-bag count, market-survival proxy, and source
  flags;
- R600, R1200, and R4800 mean, standard deviation, sample count, and fidelity
  mask;
- champion action, high-confidence winner, and canonical action hash.

The converter must reconstruct every source trajectory from its numeric seed,
verify each recorded parent public-state hash, verify every canonical action
hash and legality, verify the staged public-state hash, apply exactly the
recorded champion action, and verify the terminal state hash and score
breakdown. Any mismatch rejects the shard.

The format is append-only at whole-game boundaries, atomically written,
checksummed, resumable, and independently decoded by Rust and Python. It must
reject source, schema, split, seed, action-count, checksum, provenance, or
identity drift.

## Frozen Model

Train a fresh MLX model named `complete-action-graded-residual-v1`.

- Hidden width: 192.
- Attention heads: 6.
- Board set-attention blocks: 3.
- Market set-attention blocks: 2.
- Feed-forward multiplier: 4.
- Parent state: shared board, opponents, market, tokens, phase, and supply.
- Action path: explicit action and delta features, staged market and supply,
  and action-query cross-attention over board and staged-market tokens.
- Observable prior path: historical model immediate score, historical model
  remaining value, screen value, scaled screen rank, inverse screen rank,
  market-survival proxy, visible wildlife count, and public-bag wildlife
  count. Source flags and fidelity masks are supervision metadata and may not
  enter the model.
- Candidate-set context: masked mean and maximum pooling only; no quadratic
  action-to-action attention.
- Score: frozen screen value plus a zero-initialized learned residual.
- Residual range: `12 * tanh(raw_residual)`.
- Auxiliary output: positive predicted rollout standard error.
- Hex symmetry: one uniformly sampled exact rotation per group per epoch.
- Warm start, historical apprentice weights, hidden-state features, hidden bag
  order, and future refill order are prohibited.

The loss is frozen:

1. uncertainty-weighted Huber residual loss on every R1200 action, weight 1.0;
2. the same loss on every R4800 action, weight 4.0;
3. R1200 listwise soft-target cross-entropy at temperature 2.0, weight 0.5;
4. complete-group cross-entropy on the R4800 winner at temperature 1.0,
   weight 1.0;
5. Gaussian standard-error calibration on scored actions, weight 0.1;
6. squared residual regularization on screen-only actions, weight 0.01.

Teacher uncertainty uses `stddev / sqrt(samples)` with a fixed one-point
variance floor. Ties use the canonical action hash as the stable total order.

## Training Protocol

Train three otherwise identical replicas in parallel:

| Host | Training seed |
|---|---:|
| john1 | 2026061601 |
| john2 | 2026061602 |
| john3 | 2026061603 |

- Optimizer: AdamW.
- Learning rate: `1e-4`.
- Weight decay: `1e-4`.
- Maximum epochs: 30.
- Validation patience: 6.
- Multi-group packing target: 8,192 padded action rows.
- An indivisible larger group runs alone, with a frozen hard ceiling of 16,384
  actions; no group is split, sampled, or truncated.
- Checkpoint interval: 250 optimizer steps and every completed epoch.
- Selection objective: lowest validation retained mean R4800 regret.
- Tie breakers, in order: higher top-64 R4800-winner recall, lower R4800
  residual MAE, lower training seed.

Each host validates its local source inputs before training and writes a
manifest with device, MLX version, source identity, checkpoint hashes,
optimizer cursor, runtime, memory, and complete validation metrics. Resuming
must reproduce the uninterrupted run from the same cursor.

No architecture sweep, learning-rate sweep, extra seed, ensemble, threshold
change, calibration fit, warm start, validation partition change, or test
access is authorized.

## Validation Gates

The selected replica advances only if all integrity gates pass and:

- top-64 recall of the R4800 winner is strictly greater than 98%;
- retained mean R4800 regret is strictly less than 0.15 points;
- early, middle, and late top-64 recall are each at least 97%;
- early, middle, and late retained mean regret are each below 0.20 points;
- Nature-Token-available and independent-draft-winner subsets each achieve at
  least 95% top-64 recall and below 0.25 retained mean regret when the subset
  has at least 20 groups;
- all legal actions receive one finite score and no source action is dropped;
- warmed inference sustains at least 20,000 action scores per second on every
  Mac and P99 complete-decision scoring is at most 250 ms;
- peak process RSS is at most 4 GiB and no swap is consumed.

Report top-1/top-8/top-32/top-64 recall, retained regret, R1200/R4800 MAE and
correlation, uncertainty calibration, screen-only residual distribution,
phase and action-family slices, throughput, memory, and all integrity checks.

Missing any gate rejects this exact model before test or gameplay. A complete
pass authorizes only ADR 0082.

## Cluster Execution

All three Macs participate:

- each converts and validates the source games originally produced on it;
- john1 merges immutable train/validation/test manifests;
- all three train one MLX replica concurrently under a host-wide resource
  lock and `caffeinate`;
- each replica is cross-evaluated on a different Mac before model selection;
- the cluster dashboard records queue depth, productive time, queued idle
  time, retries, CPU, memory, and work completed.

No healthy host may remain idle for more than five minutes while compatible
conversion, validation, training, or cross-evaluation work is queued.

## Outcome

The corrected observable-only experiment completed all three authorized MLX
replicas and the frozen cross-host matrix. The selected john2 checkpoint was
`step-000003592-epoch-0008-batch-000000`.

- retained mean top-64 R4800 regret: 0.090184, passing the <0.15 gate;
- top-64 R4800-winner recall: 73.33%, failing the >98% gate;
- early/middle/late recall: 69.05% / 65.48% / 87.50%, all below 97%;
- Nature-Token and independent-draft recall: 74.35% / 71.43%, both below 95%;
- selected-model throughput: 101,888-102,405 action scores/s on all three Macs;
- P99 complete-decision scoring: 79.68-79.88 ms on all three Macs;
- all integrity, finite-score, regret, memory, and swap gates passed.

Cross-host metrics were bit-identical to each selected training checkpoint.
The selected model identity also matched on john1, john2, and john3. This is a
target-design rejection, not an execution or portability failure. The frozen
recall thresholds are not weakened after validation.

ADR 0082 was not authorized. No sealed-test group or gameplay seed was opened.
The complete machine-readable result is
`docs/v2/reports/complete-action-graded-oracle-ranker-v1-rejection.json`.

## Maximum Compute

One lossless conversion of the 13 existing games, three 30-epoch MLX replicas,
one cross-host validation pass per replica, and correctness/performance
smokes. No new teacher game, gameplay seed, external compute, fourth replica,
or reopened K2048 screen is authorized by this ADR.
