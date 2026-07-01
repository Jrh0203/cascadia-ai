# ADR 0089: Complete-Action Frontier-Anchored Set Ranker

Status: rejected on validation; sealed test and gameplay closed unopened.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-anchored-set-ranker-v1`

## Context

ADR 0088 rejected explicit rotation-canonical local geometry after the
selected model reached only 74.17% exact top-64 R4800-winner recall and
87.92% confidence-set coverage. Its 0.093757 retained regret passed, all
cross-host outputs were bit-identical, and performance passed, so execution
and model capacity are not the immediate explanation.

The validation evidence now isolates a sharper mechanism:

- all 11 R4800 winners outside the R1200 top 64 were already members of the
  deterministic champion/frontier proposal;
- a width-64 selector that unconditionally keeps those public frontier
  anchors can reserve the learned model for the ordinary nonfrontier fill;
- ADR 0088's model displaced 62 winners that R1200 already ranked inside its
  top 64, so winner-only scalar regression is a poor fit for a set-retention
  decision.

## Hypothesis

A deployable width-64 selector that always retains every deterministic
champion/frontier action and trains the unchanged observable ranker to fill
only the remaining nonfrontier slots as a set will recover the stable R4800
winner and confidence set substantially more often than scalar residual
ranking. It should do so without wider screening, hidden state, new teacher
compute, or a larger network.

## Frozen Inputs

- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- Train games: `61000-61002,61005-61006,61009-61010`.
- Validation games: `61003,61007,61011`.
- Sealed-test games `61004,61008,61012` remain prohibited.
- Existing screen, R600, R1200, and R4800 labels remain immutable.
- The proposal width remains exactly 64.
- No new rollout, label, position, game, or external compute is authorized.

## Frozen Treatment

Use the unchanged ADR 0081 `GradedOracleRanker` architecture and observable
input schema, trained from scratch.

For each decision:

1. identify every action emitted by the deterministic champion/frontier
   generator;
2. retain all such frontier actions unconditionally;
3. rank only nonfrontier actions with the MLX model; and
4. fill the remaining slots to exactly 64 using stable score order with the
   canonical action hash as the tie-break.

The training target excludes frontier actions from the learned quota. It
marks the highest-R1200 nonfrontier actions needed to fill the exact remaining
width. The loss is frozen as:

- uniform target-set cross entropy over eligible nonfrontier actions;
- plus `0.5` times temperature-2 R1200 listwise cross entropy;
- plus `0.01` times squared residual regularization on screen-only
  nonfrontier actions.

Only the `GRADED_SOURCE_CHAMPION_FRONTIER` bit may affect target construction
and final anchored selection. No source or fidelity bit enters the neural
model. Frontier membership is deterministic public candidate-generation
provenance available to the deployed selector, not hidden game information.

## Mandatory Target-Ceiling Gate

Before any training, evaluate the deterministic set containing all frontier
anchors plus the R1200 nonfrontier target fill on the already-open validation
split. All four Macs independently replay the audit and must produce
bit-identical scientific payloads and digests.

Training is authorized only if:

- exact R4800-winner recall is strictly greater than 98%;
- R4800 95% confidence-set coverage is at least 99%;
- distinguishable-winner recall is at least 98%;
- retained mean R4800 regret is below 0.03;
- early, middle, and late exact recall and confidence coverage are each at
  least 98%, with retained regret below 0.03;
- every Nature-Token or independent-draft subset with at least 20 groups
  reaches at least 95% exact recall and below 0.25 regret;
- all groups and actions are observed exactly once at width 64; and
- the sealed test remains unopened.

Any miss rejects the target before model training. Thresholds may not be
weakened after observing the ceiling.

## Correctness Gates Before Training

- Unit tests prove frontier exclusion from the learned quota, stable R1200
  target construction, unconditional anchor retention, exact width, strict
  gate boundaries, finite loss, and gradient flow to the score head.
- The frozen four-seed configuration rejects architecture, optimizer,
  dataset, warm-start, epoch, packing, and patience drift.
- A maximum-width real group on every Mac performs a finite forward pass and
  optimizer step, starts bit-exact to the historical screen, retains exactly
  64 actions, stays below 4 GiB RSS, and consumes no swap.
- Evaluator identities omit host-specific paths and reject any split other
  than validation.

## Training And Selection

Run four concurrent replicas:

| Host | Seed | Cross-replay host |
|---|---:|---|
| john1 | `2026061601` | john3 |
| john2 | `2026061602` | john4 |
| john3 | `2026061603` | john1 |
| john4 | `2026061604` | john2 |

Use AdamW, learning rate `1e-4`, weight decay `1e-4`, at most 30 epochs,
six-epoch validation patience, 8,192 packed action rows, a 16,384-action hard
group ceiling, rotation augmentation, and checkpointing every 250 optimizer
steps and completed epoch.

Within each replica and across replicas, select lowest top-64 R4800-winner
miss rate, then higher R4800 95% confidence-set coverage, lower retained mean
R4800 regret, and lower seed. Every selected replica is replayed on its frozen
cross host. Scientific rankings and metrics must be bit-identical.

## Frozen Validation Gates

The selected replica advances only if every integrity, finite-score, and
cross-host identity check passes and:

- exact top-64 R4800-winner recall is strictly greater than 98%;
- top-64 R4800 95% confidence-set coverage is at least 99%;
- distinguishable R4800-winner recall is at least 98%;
- retained mean top-64 R4800 regret is below 0.15;
- early, middle, and late exact recall are each at least 97%;
- early, middle, and late confidence-set coverage are each at least 98%;
- early, middle, and late retained regret are each below 0.20;
- Nature-Token and independent-draft subsets with at least 20 groups each
  achieve at least 95% exact recall and below 0.25 retained regret; and
- every Mac sustains at least 20,000 action scores per second, P99 decision
  scoring at most 250 ms, peak RSS at most 4 GiB, zero process swaps, and no
  positive system-swap delta.

Passing authorizes only a separately frozen sealed-test ADR. Missing any gate
rejects this treatment before test or gameplay.

## Cluster Execution

All four healthy Apple Silicon Macs receive work under host locks and
`caffeinate`. Source, environment, dataset, and experiment-manifest
identities are verified before each job. No healthy node may remain idle while
a compatible ceiling replay, smoke, replica, cross-evaluation, or performance
replay is queued.

## Maximum Compute

Four deterministic target-ceiling replays, four maximum-width smokes, four
30-epoch MLX replicas, one cross-host replay per replica, and the required
correctness and reporting work. No architecture sweep, loss sweep, optimizer
sweep, fifth seed, warm start, new teacher compute, sealed-test access,
gameplay, K2048, Modal, cloud, or other external compute is authorized.

## Result

The deterministic target ceiling passed on all four Macs at 99.58% exact
R4800-winner recall, 100% confidence-set coverage, 100% distinguishable
recall, and 0.000526 retained regret. All four maximum-width forward/backward
smokes and the full correctness suite passed, so the target and implementation
were trainable in principle.

All four frozen replicas then completed under the six-epoch patience rule.
The john2 seed won the preregistered selection order at checkpoint
`step-000003592-epoch-0008-batch-000000`, with:

- 76.67% exact top-64 R4800-winner recall;
- 90.42% R4800 95% confidence-set coverage;
- 92.11% distinguishable-winner recall;
- 0.061734 retained mean top-64 R4800 regret;
- 26.21% nonfrontier target-positive recall; and
- 0% exact nonfrontier target-set recovery.

Exact recall improved only 3.75 percentage points from the anchored screen.
Early, middle, and late recall reached 70.24%, 73.81%, and 87.50%;
Nature-Token and independent-draft recall reached 76.96% and 76.19%. Eleven
frozen quality gates failed.

Every origin/cross scientific payload and digest matched bit-for-bit. All
eight performance replays passed at 96,547-98,806 action scores per second,
P99 decision scoring stayed between 83.22 and 86.74 ms, peak RSS remained
below 502 MiB, process swaps were zero, and no replay consumed system swap.
The scoped 85-file MLX runtime source bundle was byte-identical across all
four Macs. Twenty host-locked jobs completed with no failure or retry. Sealed
test, gameplay, new teacher compute, K2048, and external compute remained
closed.

The hypothesis is rejected. Hard frontier retention is mechanically sound,
but the unchanged observable ranker did not learn the required nonfrontier
set allocation. The next research cycle must distinguish representation
collision, optimization pressure, and generalization failure before another
training treatment. Under the revised cluster policy, those diagnostics run
as independent experiments across the four Macs; duplicate training replicas
are reserved for confirmation after a single-host pilot passes.

Machine-readable closure:
`docs/v2/reports/complete-action-frontier-anchored-set-ranker-v1-rejection.json`.
