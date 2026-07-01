# ADR 0156: R4 Bounded Quotient MLX Comparison

Status: completed; invalid; no representation selected

Date: 2026-06-17

Experiment: `r4-bounded-quotient-mlx-comparison-v1`

Protocol: `r4-bounded-parent-mlx-matched-comparison-v1`

Research-plan item: R4 learned successor

Foundation: ADR 0155 and
`docs/v2/reports/r4-bounded-far-quotient-foundation-v1-result.md`

## Context

ADR 0155 admitted three hard-bounded radius-four model views:

| Arm | P99 tokens | Paired construction throughput |
|---|---:|---:|
| Q1 seat-marginal | 166 | 1.209x full HWF |
| Q2 directional | 186 | 1.163x full HWF |
| Q3 habitat-affordance | 182 | 1.121x full HWF |

All three retain the accepted 61-cell focal field, exact far habitat and
wildlife components, exact `CSR4AM1` overflow sidecar, all seven registered
long-range distinctions, and exact wildlife/frontier source accounting.

That mechanical result does not establish learned sufficiency. The next
question is:

> Can a bounded R4 parent-state context replace the exact R2 parent context in
> the accepted complete-action ranker without losing decision quality, while
> reducing realistic parent-state serving cost enough to matter?

The comparison must not regenerate the 8.3 GiB R3 candidate cache, conflate
state compression with candidate-action compression, or compare performance
across unlike Macs. The exact R2 candidate afterstate stream therefore remains
common across all arms. Only the once-per-decision parent representation
changes.

## Decision

Run four iso-model, iso-data MLX arms concurrently:

| Arm | Parent representation | Host |
|---|---|---|
| `c0-exact-r2-parent` | exact full R2 sparse parent | john1 |
| `q1-seat-marginal-parent` | ADR 0155 Q1 | john2 |
| `q2-directional-parent` | ADR 0155 Q2 | john3 |
| `q3-affordance-parent` | ADR 0155 Q3 | john4 |

Every arm uses the same:

- 560 open train decisions and deterministic at-most-512 action cohorts;
- 240 complete open validation decisions and 860,203 legal actions;
- exact R2 active-board candidate afterstate tokens;
- graded action features, priors, staged market, exact semantic supply, and
  frontier compatibility;
- R600/R1200/R4800 labels;
- model module graph and trainable parameter count;
- initial parameter tensor;
- optimizer, loss, training steps, candidate sampler, and D6 schedule;
- validation metrics and stable action-hash tie breaking; and
- isolated serving benchmark protocol.

The arm identifier routes only the parent-state loader. It does not alter the
candidate encoder, factual inputs, objective, output heads, or training
budget.

## Why The Parent Boundary

The current ranker already factorizes a decision into:

1. one reusable parent-state encode; and
2. one exact active-board afterstate encode per candidate.

Replacing only the parent isolates the R4 state hypothesis. It also keeps the
candidate action surface exact and common, avoiding the focal-seat ambiguity
that would arise if a post-turn R4 afterstate automatically recentered on the
next player.

A passing arm therefore selects a compact parent substrate for complete-action
ranking. It does not establish that a bounded R4 view can replace every
candidate afterstate or every future search leaf.

## Exact Parent Sidecar

Build one content-addressed Rust-authored sidecar aligned to the accepted R3
cache:

- R3 cache:
  `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156`;
- S1 cache:
  `2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15`;
- train groups: 560;
- validation groups: 240.

For every group and Q1/Q2/Q3 arm, Rust:

1. verifies the source group and public-state hash against the R3 cache;
2. constructs exact R2 and `CSR4AM1`;
3. constructs the bounded view;
4. validates exact decode, source accounting, and the ADR 0155 hard maximum;
5. carries the exact center through all twelve D6 transforms;
6. validates each transform/inverse and bounded envelope; and
7. writes only Rust-authored token kinds, relative seats, active `i16`
   values, offsets, counts, and scientific identities.

The sidecar stores all twelve transformed bounded views. Python selects the
precomputed transform fixed by the common training schedule and never
reimplements sector, radial, component, frontier, or summary transforms.

The sidecar is ragged. It stores active scalar values rather than a
`records x transforms x boards x 204 x 144` dense tensor. Loading may pad only
the rows selected for the current batch. Truncation is prohibited.

## Universal Parent Token Contract

The matched parent encoder accepts nine semantic token classes:

1. R2 occupied;
2. R2 legal frontier;
3. R2 habitat component;
4. R2 wildlife motif;
5. Q near cell;
6. Q far habitat component;
7. Q far wildlife component;
8. Q wildlife quotient summary; and
9. Q frontier quotient summary.

Every token has:

- one class code;
- one explicit relative-seat owner;
- up to 144 schema-positioned signed integer values; and
- one active mask.

R2 payloads are zero-extended from 52 values. Q payloads use their exact
ADR 0155 active values; no field is hashed, clipped, pooled, or silently
discarded.

The common MLX adapter contains one registered projection per semantic token
class. All four models instantiate the complete nine-class schema and have
the same parameter layout. These adapters are semantic schema components,
not count-matching ballast: every adapter is exercised by at least one
registered arm, and no arm-specific module is constructed.

Within each relative board:

- active tokens are grouped by class in canonical order;
- nine fixed type-summary tokens use independent masked means;
- one player token and sixteen fixed Perceiver latents are used;
- one cross-attention and one latent self-attention block are applied; and
- only board summaries cross boards through the common global/market/player
  context block.

The candidate encoder and all downstream heads are the accepted ADR 0150
module graph.

## Frozen Optimization

| Variable | Value |
|---|---:|
| Seed | `2026061710` |
| Optimizer | AdamW |
| Steps | 3,000 |
| Groups per step | 4 |
| Candidates per sampled group | at most 512 |
| Learning rate | `0.0001` |
| Weight decay | `0.0001` |
| Checkpoint interval | 250 |
| Metric interval | 100 |
| Validation probe | fixed 24 open groups |
| Full validation | once after step 3,000 |
| Candidate chunk | 256 |
| Initialization | fresh and byte-identical across arms |
| Warm start | prohibited |
| Early stopping | prohibited |
| MLX cache limit | 1 GiB |

The group sampler, protected-slice oversampling, D6 IDs, and objective are
byte-identical to ADR 0150 apart from the new experiment seed.

## Shared Objective

```text
r1200 uncertainty-weighted Huber
+ 4.0 * r4800 uncertainty-weighted Huber
+ 0.5 * r1200 listwise cross entropy
+ 1.0 * r4800 winner cross entropy
+ 0.1 * standard-error calibration
+ 0.01 * screen-only residual regularization
```

No reconstruction, token-count, or representation-identification auxiliary
loss is allowed.

## Quality Evaluation

Every arm reports:

- R4800 MAE, RMSE, bias, correlation, and calibration;
- top-1, top-8, top-32, and top-64 stable-winner recall;
- retained R4800 regret at the same widths;
- top-64 95% teacher-confidence-set coverage;
- early, middle, late, low-supply, and independent-draft-winner slices;
- parent and candidate token distributions;
- a fixed 64-action prediction panel; and
- complete group/action coverage.

The control must first establish a valid experiment:

```text
R4800 MAE <= 1.42
R4800 RMSE <= 1.85
top-64 winner recall >= 0.70
top-64 retained regret <= 0.12
low-supply top-64 recall >= 0.88
independent-draft-winner top-64 recall >= 0.76
top-64 confidence-set coverage >= 0.97
```

These bounds are prospective sanity limits around the prior exact-R2 control,
not promotion thresholds for treatments.

A bounded arm is quality-noninferior only if all hold:

```text
R4800 MAE delta versus C0 <= 0.05
R4800 RMSE delta versus C0 <= 0.05
top-64 winner-recall delta versus C0 >= -0.005
top-64 retained-regret delta versus C0 <= 0.005
low-supply top-64 recall delta versus C0 >= -0.01
independent-draft-winner top-64 recall delta versus C0 >= -0.01
top-64 confidence-set coverage >= 0.99
```

## Serving Evaluation

Every arm must pass:

```text
complete validation coverage == 240 decisions and 860,203 actions
finite scores and uncertainties == 100%
parent encodes == 240
process swap == 0
peak active MLX memory <= 4 GiB
peak process RSS <= 4 GiB
P99 complete-decision latency <= 250 ms
fixed-chunk action throughput >= 20,000 scores/second
```

After training, the exact C0 checkpoint is replayed as a serving-only control
on john2, john3, and john4. Each treatment is compared with that control on
the same host, cache, rows, warmup count, steady count, and system state.
Cross-host raw timing is never used as a promotion ratio.

A bounded arm is materially more efficient only if:

```text
parent-encode P50 latency <= 0.80 * host-paired C0
```

and at least one end-to-end condition holds:

```text
fixed-chunk action throughput >= 1.05 * host-paired C0
complete-decision P99 latency <= 0.95 * host-paired C0
peak active MLX memory <= 0.85 * host-paired C0
peak process RSS <= 0.85 * host-paired C0
```

## Classification

Possible classifications:

- `r4_bounded_parent_mlx_invalid`;
- `r4_bounded_parent_mlx_control_failed`;
- `r4_bounded_parent_mlx_all_treatments_degraded`;
- `r4_bounded_parent_mlx_quality_only_null`; or
- `r4_bounded_parent_mlx_representation_selected`.

A representation is selected only when it passes every absolute and quality
gate and the complete material-efficiency rule.

Among eligible arms, select the highest host-paired fixed-chunk throughput
ratio. Ratios within 1% tie-break by lower host-paired complete-decision P99,
then lower peak active memory, then Q1, Q3, Q2 order. The final order favors
the smaller registered quotient when performance is practically tied.

## Cluster Execution

The production graph must include:

1. immutable source/exporter bundle;
2. whole-tree bundle fanout and verification on all Macs;
3. one exact parent-sidecar export on john1;
4. checksum-bound sidecar fanout;
5. four host/arm MLX preflights;
6. a bounded numerical-parity smoke on john1 and john4;
7. four concurrent 3,000-step training runs;
8. exact C0 checkpoint fanout after training;
9. three host-paired C0 serving replays;
10. arm report collection;
11. forward and reverse classification; and
12. byte-identical order proof.

Training uses one MLX slot per Mac. Cache export, fanout, collection, and
classification are shared prerequisites and do not duplicate scientific
training.

The smoke arm, prefix bound, tolerances, and claim boundary are frozen by
`docs/v2/reports/r4-bounded-parent-mlx-cross-host-smoke-amendment-2026-06-17.md`.
The all-validation-row parent-latency population is frozen by
`docs/v2/reports/r4-bounded-parent-mlx-serving-parent-latency-amendment-2026-06-17.md`.
The post-training replay coverage and artifact-isolation correction is frozen
by
`docs/v2/reports/r4-bounded-parent-mlx-control-replay-amendment-2026-06-17.md`.

## Claim Boundary

Passing may select a compact parent-state substrate for subsequent
complete-action modeling and paired gameplay.

It cannot establish a pure bounded candidate-afterstate representation,
search strength, gameplay strength, or progress toward the 100-point mean
target.

Any change to arms, parent/candidate boundary, token schema, source caches,
data, labels, model graph, seed, optimizer, schedule, objective, host
assignment, smoke tolerances, or promotion gates requires an ADR amendment
before production.

## Outcome

All four registered arms completed exactly 3,000 optimizer steps and scored
all 240 validation decisions and 860,203 actions exactly once. The formal
classifier returned:

```text
r4_bounded_parent_mlx_invalid
selected_arm = null
```

The invalidating condition was the exact C0 same-host replay on john2, which
peaked at 4.1456 GiB process RSS against the frozen 4 GiB absolute serving
limit. Forward and reverse classification were byte-identical.

The remaining evidence also rules out a useful promotion from this run:

- C0 missed the frozen MAE, RMSE, and low-supply recall sanity limits;
- Q1, Q2, and Q3 each missed multiple quality-noninferiority gates;
- no treatment reached the mandatory `0.80x` same-host parent-encode P50
  ratio; and
- Q1 and Q3 themselves exceeded the 4 GiB process-RSS limit.

The threshold was not changed after observing the result, and the replay was
not repeated until it happened to pass. No R4 bounded parent representation is
selected. Exact sparse R2 remains the accepted learned candidate substrate
from ADR 0150, while the exact R4 codecs and quotient extractors remain
available for diagnostics and future ablations.

Full result:
`docs/v2/reports/r4-bounded-quotient-mlx-comparison-v1-result.md`.
