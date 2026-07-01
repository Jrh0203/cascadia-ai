# Opportunity Cross-Attention MLX Tournament V1 Preregistration

Date: 2026-06-17

ADR: 0166

Experiment: `opportunity-cross-attention-mlx-tournament-v1`

Protocol: `exact-r2-opportunity-query-factorial-v1`

Status: frozen before feasibility smoke and production

## Question

Does candidate-conditioned retrieval from exact semantic supply and exact
occupied/frontier/component/motif memory improve complete-action ranking beyond
both an identically sized parent-query adapter and the untouched exact-R2 C0
warm start?

## Representation Boundary

The experiment never materializes a 441-cell board tensor. It uses:

- sparse exact-R2 occupied, frontier, component, and motif tokens;
- exact R2 public candidate action-edit tokens;
- 80 exact-S1 supply tokens;
- the current public market, players, and global state; and
- no hidden order, excluded tile identity, future refill, sealed examples, or
  teacher-derived input feature.

An exact centered 121-cell hex disk is not a valid geometry. The nearest
centered disks contain 91 and 127 cells. R2 avoids that arbitrary crop choice
by materializing only active sparse objects.

## Factorial Arms

```text
C0: parent query -> supply; parent query -> frontier
T1: candidate query -> supply; parent query -> frontier
T2: parent query -> supply; candidate query -> frontier
T3: candidate query -> supply; candidate query -> frontier
```

All four graphs execute both attention modules. No arm receives fewer
parameters, layers, memories, or operations because of its assignment.

## Warm Start And Freeze

The warm start must be the final 3,000-step authorized C0 exact-R2 report from
ADR 0161.

Required checks:

```text
report scientific identity valid
report checkpoint hashes valid
latest checkpoint = reported checkpoint
global step = 3,000
base arm = c0-exact-r2
zero-init adapter predictions exactly equal base predictions
base tensor hash identical across all arms
base tensor hash unchanged after every production run
```

Every inherited encoder, candidate trunk, residual head, and uncertainty head
is frozen. Only the opportunity memory projections, query projections,
cross-attention modules, context trunk, and context residual are trainable.

## Frozen Corpus And Optimizer

```text
open train: 560 complete decisions, 280,012 retained actions
open validation: 240 complete decisions, 860,203 retained actions
training steps: 2,000
groups per step: 4
candidate cap per group: 512
optimizer: AdamW
learning rate: 1e-4
weight decay: 1e-4
seed: 2026061718
checkpoint interval: 250
live probe interval: 100
early stopping: false
```

The live 24-group probe is monitoring only. It cannot select an arm.

## Common Smoke

Before authorization, all four hosts run the same parent-conditioned arm from
the same immutable bundle and C0 checkpoint for the same bounded smoke steps.

The cross-host smoke must prove:

- common source, cache, C0, graph, and initial tensor identities;
- exact common scientific batch hashes;
- exact zero-init prediction parity;
- common step-one loss and prediction panel;
- frozen-base identity after smoke;
- finite optimization and evaluation;
- R6 exact apply/undo parity; and
- bounded smoke completion on every host.

The smoke is excluded from production.

## Full Validation Metrics

Primary utility:

- top-64 R4800 winner recall over all 240 decisions.

Primary mechanism:

- mean top-64 R4800 winner recall over Elk, Salmon, and Hawk opportunity
  subsets.

Protected:

- low-supply top-64 winner recall;
- independent-draft-winner top-64 winner recall;
- early, middle, and late phase recall; and
- Bear opportunity recall as a diagnostic.

Secondary:

- top-1, top-8, and top-32 winner recall;
- top-64 confidence-set coverage;
- top-64 retained R4800 regret;
- R4800 RMSE, MAE, bias, and calibration;
- model and combined complete-decision latency;
- action scores per second;
- active MLX memory, peak process RSS, and swap delta.

## Paired Evidence

The classifier reconstructs one stable per-decision record for every arm:

```text
teacher winner retained at K=64
retained R4800 regret
teacher winner predicted rank
mean absolute and squared R4800 error
phase and protected-slice memberships
Elk, Salmon, Hawk, and Bear opportunity memberships
```

For every candidate-query treatment, it reports paired differences against
both the parent-conditioned adapter and untouched C0. Bootstrap resampling uses
complete decisions as the unit, a frozen seed, and 100,000 replicates.

Each production report persists its complete 240-decision panel and panel
BLAKE3. Collection also retrieves the final `checkpoint.json` and
`model.safetensors`; the classifier rejects any report whose claimed
checkpoint bytes do not match those collected artifacts.

The report also computes the two-by-two factorial:

```text
supply main effect = ((T1 - C0) + (T3 - T2)) / 2
frontier main effect = ((T2 - C0) + (T3 - T1)) / 2
interaction = T3 - T1 - T2 + C0
```

## Advancement Gates

A treatment is structurally valid only if all source, data, warm-start,
parameter, batch, report, and R6 identities pass.

A structurally valid treatment advances only when:

```text
absolute serving:
  combined complete-decision P99 <= 250 ms
  peak process RSS <= 4 GiB
  system swap delta <= 0

global utility:
  top-64 winner recall > parent-conditioned adapter
  top-64 winner recall > untouched C0
  paired bootstrap P(delta > 0) >= 0.95 versus parent-conditioned adapter

mechanism:
  strategic-opportunity recall > parent-conditioned adapter
  strategic paired bootstrap P(delta > 0) >= 0.90

protected noninferiority:
  low-supply recall delta >= -0.02
  independent-draft recall delta >= -0.02

value and regret noninferiority:
  R4800 RMSE delta <= +0.03
  mean top-64 retained-regret delta <= +0.02
```

All deltas above use the parent-conditioned adapter unless explicitly stated.
Untouched C0 remains an additional absolute utility control.

If no arm passes every gate, the experiment is null or negative even if one
arm leads a table.

## Classification

```text
opportunity_query_candidate_advance
opportunity_parent_context_only
opportunity_query_factorial_null
opportunity_query_quality_regression
opportunity_query_protected_slice_regression
opportunity_query_serving_failure
opportunity_query_structurally_invalid
opportunity_query_cross_host_inconsistent
```

## Machine Allocation

| Host | Production arm | Overlap |
|---|---|---|
| john1 | C0 parent-conditioned | after local S6 shard |
| john2 | T1 supply query | after ADR 0161 paired C0 replay |
| john3 | T2 frontier query | after ADR 0161 paired C0 replay |
| john4 | T3 combined query | after ADR 0161 paired C0 replay |

No host repeats another production arm. The staggered schedule is deliberate:
it maximizes cluster utilization without sacrificing the same-host controls
needed to close ADR 0161.

The frozen scheduler graph contains 20 tasks and is emitted by
`tools/opportunity_cross_attention_mlx_queue.py`. It requires:

- the completed exact C0 run fanout;
- the john1 S6 shard before john1 smoke;
- each john2 through john4 ADR 0161 paired replay before that host's smoke;
- a passing four-host smoke before authorization;
- all four production preflights before any production optimizer;
- one untouched-C0 decision panel before the john1 parent arm;
- exact collection of all four reports and final checkpoint bytes; and
- one order-invariant decision-terminal classification.

The same tool builds the immutable, content-addressed source and R6 bundle from
an explicit include list. Bundle validation and whole-tree checksum fanout
precede every host smoke.

Queue revision `v2` uses absolute john1 paths for every fanout source,
collection destination, and report output. Tasks run from the immutable
bundle's `source/` directory, so repository-relative artifact paths are an
invalid orchestration state. Revision `v1` reached no scientific compute; its
failed first fanout remains preserved as evidence.

## Gameplay Ladder

Offline selection authorizes:

1. 50 paired games split across four hosts;
2. 200 paired games if the first gate is positive;
3. component score and decision-latency analysis; and
4. only then a larger score-to-100 qualification.

The offline tournament cannot claim gameplay improvement or progress to 100.

## Invalidators

- production before this preregistration;
- any 441-cell dense input or arbitrary claim of an exact 121-cell disk;
- a different C0 checkpoint across hosts;
- updating inherited parameters;
- unequal arm graphs, optimizer schedules, batches, or labels;
- selecting from the live probe;
- opening sealed data;
- hidden-order or future-refill leakage;
- omitting R6 cost from serving;
- report-order-dependent classification; or
- declaring the leading arm successful after it fails a gate.
