# ADR 0161: Relational Substrate MLX Tournament

Status: accepted; implementation in progress

Date: 2026-06-17

Experiment: `relational-substrate-mlx-tournament-v1`

Protocol: `r5-s3-s5-matched-mlx-v1`

Research-plan items: R5, R6, S3, S5

## Context

The representation foundations now establish four facts:

1. exact sparse R2 is a strong learned control and does not require a
   historical 441-cell lattice;
2. R5 component-and-motif state plus a small action-local patch is exact for
   the tested Card A score and affordance decoders while using 59.57% of the
   control's median parent tokens;
3. S3 component, motif, frontier, and opportunity objects are exact and D6
   consistent; and
4. S5 exposes an exact 154-field target-free action derivative whose raw
   scales require explicit normalization.

R6 also demonstrated exact incremental apply/undo at 58.86x the measured
authoritative reconstruction time. Learned quality, protected-slice ranking,
and end-to-end serving value remain unknown.

The earlier R3 comparison is an important control. A generic variable-token
local action encoder was slower and value-inferior to exact R2. This
tournament does not repeat that architecture unchanged. It asks whether
explicit quotient, topology, and counterfactual structure can make the compact
action path useful under one matched MLX graph.

## Decision

Run one four-arm, capacity-matched MLX tournament:

| Arm | Parent | Candidate | Additional factual input | Host |
|---|---|---|---|---|
| `c0-exact-r2` | exact sparse R2 | exact full R2 afterstate | none | john1 |
| `q1-r5-quotient-local` | score-sufficient R5 component/motif quotient | exact R3 radius-one plus global edit | none | john2 |
| `g2-r5-s3` | R5 quotient plus rich S3 component, motif, frontier, and opportunity fields | exact R3 radius-one plus global edit | none | john3 |
| `d3-r5-s3-s5` | same rich S3 parent | exact R3 radius-one plus global edit | normalized exact S5 derivative | john4 |

All candidate cohorts, public parent states, authoritative afterstates,
action hashes, labels, and D6 schedules come from the immutable accepted R3
cache. "Same candidate afterstates" means that every arm sees the same
retained source action and authoritative factual successor. The compact arms
consume the already-verified R3 local/global encoding of that successor
instead of silently reconstructing a different action set.

## Shared Model Graph

Every arm instantiates the same trainable graph:

- the accepted R2 token projection for exact R2 tokens;
- one class-specific 64-value relational token adapter;
- 16 fixed parent Perceiver latents;
- one parent latent block and one cross-board block;
- the accepted eight-latent R3 candidate Perceiver;
- the same action, prior, market, exact-supply, staged-supply, archetype, and
  frontier adapters;
- one 154-field derivative adapter; and
- one common fusion and output head.

Absent surfaces are represented by zero factual inputs or masked token
classes. No arm removes parameters. Parameter count, parameter layout,
initial tensor hash, optimizer, steps, and loss must match exactly.

The exact R2 control retains its original token projection rather than routing
through the failed ADR 0156 universal-value approximation.

## Relational Token Contract

The sidecar stores one rich graph and derives the R5-minimal view by masking
fields, avoiding duplicate caches.

Token classes:

1. habitat component;
2. Bear component;
3. Elk line;
4. Salmon component;
5. Hawk position;
6. Fox center;
7. frontier summary; and
8. opportunity summary.

Each token has a relative seat and 64 signed integer value slots. The R5 view
keeps classes 1-6 and only score-sufficient geometry. The S3 view keeps all
classes and rich topology, continuation, conflict, frontier, and opportunity
fields. Exact R2 remains four separate native token types and is not
lossily converted into the relational schema.

No token may be clipped or silently truncated. The cache records observed
capacities and fails closed if a value does not fit its declared integer
domain.

## S5 Normalization

The sidecar exports all 154 raw signed fields for every retained train and
validation action. Normalization is fitted only on the open train cohort:

- all-zero field: identity with divisor one;
- ordinary field: divide by `max(train P99 absolute, 1)`; and
- heavy-tail field: signed `log1p`, then divide by the same robust divisor
  when train maximum absolute exceeds 16 times train P99 absolute.

Validation values never select a transform or divisor. Teacher values,
rollout labels, hidden refill order, and terminal targets never participate.

## Frozen Data And Optimization

The tournament reuses:

- R3 cache
  `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156`;
- S1 cache
  `2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15`;
- 560 train decisions with the exact R3 at-most-512 cohort; and
- all 240 validation decisions and 860,203 validation actions.

Training remains 3,000 AdamW steps, four groups per step, learning rate
`1e-4`, weight decay `1e-4`, no warm start, no early stopping, and the frozen
ADR 0150 graded-oracle objective.

## Quality Gates

The control must satisfy the existing strong-control envelope:

```text
R4800 MAE <= 1.42
R4800 RMSE <= 1.85
top-64 winner recall >= 0.70
top-64 retained regret <= 0.12
low-supply top-64 recall >= 0.88
independent-draft top-64 recall >= 0.76
confidence-set coverage >= 0.97
```

Every treatment must satisfy all matched noninferiority limits:

```text
MAE delta <= +0.05
RMSE delta <= +0.05
top-64 recall delta >= -0.005
top-64 regret delta <= +0.005
low-supply recall delta >= -0.01
independent-draft recall delta >= -0.01
confidence-set coverage >= 0.99
```

It must also improve the mean top-64 recall over the Elk-extension,
Salmon-continuation, and Hawk-opportunity subsets by at least 0.015, with no
individual subset worse than control by more than 0.01.

## Serving Gates

Fresh-process serving reports:

- fixed-chunk model-only scores per second;
- parent encode latency;
- complete-decision model latency;
- cache materialization latency;
- exact R6 apply/undo latency and parity;
- combined complete-decision latency and actions per second;
- MLX active, cache, and peak memory;
- process RSS and process/system swap; and
- token and derivative materialization distributions.

Every arm must score at least 20,000 actions per second model-only, remain
below 250 ms combined complete-decision P99, remain below 4 GiB active and
process RSS, and record zero process swaps. A treatment is materially more
efficient only if its host-paired exact-control replay improves combined
actions per second by at least 10% or reduces combined P99 by at least 10%.

R6 apply/undo must match the authoritative parent digest for every replayed
action. A model result without R6 parity is invalid, not merely slow.

## Selection

Classification precedence:

1. invalid evidence;
2. control failed;
3. all treatments degraded;
4. quality-only null;
5. relational substrate selected.

A treatment is eligible only if quality noninferiority, strategic-slice gain,
absolute serving, R6 parity, and material efficiency all pass. Selection then
orders eligible treatments by:

1. highest mean strategic-slice recall gain;
2. lowest top-64 retained regret;
3. highest combined complete-decision throughput;
4. lowest peak process RSS; and
5. simpler arm order `q1`, `g2`, `d3`.

The tournament selects one primary substrate and records the best
quality-noninferior nonselected arm as the fallback. No gameplay run is
authorized unless a treatment is selected.

## Consequences

A pass promotes one compact substrate into the integrated relational policy
campaign and authorizes a paired gameplay qualification. A null result keeps
exact R2 as the accepted substrate and preserves S3/S5 only as narrower
feature hypotheses.

Regardless of outcome, 441 padded cells remain closed. Exact sparse
coordinates, 91/127-cell bounded controls with overflow, and semantic
quotients remain the only authorized spatial directions.

## Claim Boundary

This experiment can establish offline ranking quality and serving efficiency.
It cannot by itself claim a gameplay-score gain, a new champion, or a
100-point mean.
