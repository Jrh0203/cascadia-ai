# ADR 0088: Complete-Action Local-Geometry Ranker

Status: rejected on validation; sealed test and gameplay closed unopened.

Date: 2026-06-16

Experiment ID: `complete-action-local-geometry-ranker-v1`

## Context

ADR 0081's observable complete-action ranker reduced retained R4800 regret but
recalled the exact R4800 winner in only 73.33% of validation top-64 sets. ADR
0086 showed that finite-sample near-ties do not explain the full miss rate.
ADR 0087 then established a 95.42% exact-winner and 97.08% confidence-set
coverage upper bound from stable R1200 ordering, while proving that every
selected-model top-64 action was already R1200-labeled.

A post-ADR diagnostic, using only the already-open validation split, found:

- all 11 R4800 winners below R1200 rank 64 were champion/frontier anchors;
- the selected model recovered 9 of those 11 anchor exceptions;
- the same model displaced 62 R4800 winners that R1200 already ranked inside
  its top 64.

The dominant observed gap is therefore not unlabeled-action displacement or
failure to recognize the known anchor exceptions. The current architecture
must infer candidate-to-board hex relations indirectly from independently
projected absolute coordinates. This is unnecessarily sample-inefficient for
habitat edge matching and Bear, Elk, Salmon, Hawk, and Fox local structure.

## Hypothesis

An explicit, observable, rotation-canonical local geometry path will improve
complete-action ranking by making the exact board neighborhood modified by an
action directly available to the model. It should recover ordinary
R1200-qualified winners without depending on source flags, fidelity masks,
future bag order, hidden state, or additional teacher compute.

## Frozen Inputs

- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- Train games: `61000-61002,61005-61006,61009-61010`.
- Validation games: `61003,61007,61011`.
- Sealed-test games `61004,61008,61012` remain prohibited.
- The grouped dataset, screen prior, R600/R1200/R4800 labels, exact action
  deltas, augmentation, objective, optimizer, batch packing, epoch budget,
  patience, and checkpoint-selection order remain identical to ADR 0081.
- Training seeds remain `2026061601`, `2026061602`, and `2026061603`, paired
  by host with ADR 0081.

## Frozen Treatment

Train `complete-action-graded-local-geometry-v1` from scratch. It contains the
entire ADR 0081 base ranker plus one additive local-geometry residual path.
The base ranker is instantiated first so paired seeds reproduce its original
initial tensors exactly.

For each candidate, the new path observes only the active board and constructs
13 exact relation slots:

1. the six existing tiles adjacent to the candidate tile coordinate;
2. the existing tile at the candidate wildlife coordinate;
3. the six existing tiles adjacent to the candidate wildlife coordinate.

The six directions are expressed in the candidate tile's rotation-local
frame. Selected board entities omit absolute coordinates and replace absolute
tile rotation with rotation relative to the candidate tile. The local action
path likewise omits absolute tile/wildlife coordinates and absolute rotation.
Thus the added path is exactly invariant to a joint 60-degree rotation of the
position and action.

The 13 relation entities, relation-presence bits, canonical action features,
observable screen priors, and public global features are projected to one
candidate embedding. Masked candidate mean and maximum context feed a bounded
additive residual correction. Its final linear head is zero-initialized, so
the untrained treatment is exactly the ADR 0081 screen-initialized model.
The combined residual remains clipped to the existing `[-12, 12]` contract.

Teacher source flags, fidelity masks, selected/champion identities, rollout
means, rollout variances, and sample counts may be used only by the unchanged
loss and evaluator. They may not enter the model input.

## Correctness Gates Before Training

- Unit tests prove exact relation-slot occupancy and padding behavior.
- Jointly rotating a board and action through all six symmetries produces
  bit-identical local-geometry features.
- With paired initialization and a zero local head, treatment scores,
  residuals, and standard errors exactly match the ADR 0081 base model.
- The existing loss is finite and gradients reach the new correction head.
- A real maximum-width group scores every action once with finite outputs.
- Dataset, split, teacher, source, model-schema, and sealed-test boundaries
  reject drift.

## Training And Selection

Run one replica on each Mac:

| Host | Seed |
|---|---:|
| john1 | `2026061601` |
| john2 | `2026061602` |
| john3 | `2026061603` |

Use AdamW, learning rate `1e-4`, weight decay `1e-4`, at most 30 epochs,
six-epoch validation patience, 8,192 packed action rows, a 16,384-action hard
group ceiling, and checkpointing every 250 optimizer steps and completed
epoch. Select exactly as ADR 0081: lowest validation top-64 retained R4800
regret, then higher exact top-64 R4800-winner recall, lower R4800 residual
MAE, and lower seed.

Each selected replica is replayed on another Mac. Scientific rankings and
metrics must be identical across hosts.

## Frozen Validation Gates

The selected replica advances only if every ADR 0081 integrity, performance,
memory, and finite-score gate passes and:

- exact top-64 R4800-winner recall is strictly greater than 98%;
- top-64 R4800 95% confidence-set coverage is at least 99%;
- distinguishable R4800-winner recall is at least 98%;
- retained mean top-64 R4800 regret is below 0.15 points;
- early, middle, and late exact top-64 recall are each at least 97%;
- early, middle, and late confidence-set coverage are each at least 98%;
- early, middle, and late retained regret are each below 0.20 points;
- Nature-Token and independent-draft subsets with at least 20 groups each
  achieve at least 95% exact recall and below 0.25 retained regret;
- warmed inference sustains at least 20,000 action scores per second, P99
  decision scoring is at most 250 ms, peak RSS is at most 4 GiB, and no swap
  is consumed on every Mac.

Passing authorizes only a separately frozen sealed-test ADR. Missing any gate
rejects this exact treatment before test or gameplay. Thresholds may not be
weakened after validation.

## Cluster Execution

All three healthy Macs train concurrently under the host lock and
`caffeinate`. Each host validates immutable dataset and source identities
before training. No healthy node may remain idle while a compatible replica,
cross-evaluation, or performance replay is queued.

## Maximum Compute

Three 30-epoch MLX replicas, one cross-host validation replay per selected
replica, correctness tests, and maximum-width/performance smokes. No new
teacher rollout, fourth replica, architecture sweep, loss change, optimizer
change, sealed-test access, gameplay seed, K2048 screen, or external compute
is authorized.

## Pre-Evaluation Contract Correction

The first john1/john3 cross-evaluation launch exited before loading a model or
reading a validation group. The evaluator checked `run["training"]["kind"]`,
but the shared ranking run manifest stores the adapter identity at
`run["kind"]`. The failed launches produced no scientific or performance
report. The assertion path was corrected to the actual versioned run-manifest
contract; model tensors, dataset inputs, metrics, gates, and all frozen
experimental choices are unchanged. Corrected launches use separate event
logs, and the failed events remain preserved as invalid provenance.

The first completed origin/cross replay then exposed a second provenance-only
issue: the scientific digest included the host-specific absolute dataset path.
The rankings and metrics were valid and matched, but their digests could not
be compared across Macs. Those reports are retained as invalid cross-host
identity evidence. The digest payload now records only canonical dataset ID,
split, seeds, counts, and manifest checksum. The evaluator, models, metrics,
gates, and datasets are otherwise unchanged, and all origin/cross replays are
repeated under the corrected digest contract.

## Result

All three frozen replicas completed. The john2 replica won the preregistered
selection order at checkpoint
`step-000004045-epoch-0009-batch-000000`, with:

- 74.17% exact top-64 R4800-winner recall;
- 87.92% R4800 95% confidence-set coverage;
- 88.16% distinguishable-winner recall;
- 0.093757 retained mean top-64 R4800 regret; and
- 1.721006 R4800 residual MAE.

The treatment improved exact recall by only 2.50 percentage points over the
71.67% historical screen and reduced retained regret by 17.0%. Every overall
winner-recovery gate failed. Early, middle, and late exact recall were
72.62%, 69.05%, and 81.94%; their confidence-set coverage was 92.86%,
82.14%, and 88.89%. Nature-Token and independent-draft exact recall were
75.39% and 76.19%.

Every origin/cross scientific payload and digest matched bit-for-bit. All six
performance replays passed, ranging from 43,836 to 82,249 action scores per
second with P99 decision scoring from 97.86 to 181.11 ms, peak RSS below
574 MiB, zero process swaps, and no positive system-swap delta. The three
training launches started within 0.247 seconds. john2 resumed once after its
SSH wrapper disconnected; the frozen checkpoint, source, dataset, and
scientific contracts were unchanged.

The hypothesis is rejected. Explicit local geometry is useful for regret but
does not solve complete-action winner recovery. Sealed test, gameplay, new
teacher compute, and K2048 remain unopened. The machine-readable closure is
`docs/v2/reports/complete-action-local-geometry-ranker-v1-rejection.json`.
