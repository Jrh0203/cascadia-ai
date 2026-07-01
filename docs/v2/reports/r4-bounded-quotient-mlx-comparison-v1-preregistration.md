# R4 Bounded Quotient MLX Comparison V1 Preregistration

Date: 2026-06-17

ADR: 0156

Experiment: `r4-bounded-quotient-mlx-comparison-v1`

Protocol: `r4-bounded-parent-mlx-matched-comparison-v1`

Status: preregistered; implementation evidence not yet admitted

## Question

Can Q1, Q2, or Q3 replace exact R2 as the once-per-decision parent context in
the accepted complete-action ranker without losing aggregate or protected
slice quality, while producing a material same-host serving gain?

## Frozen Arms

| Host | Arm |
|---|---|
| john1 | `c0-exact-r2-parent` |
| john2 | `q1-seat-marginal-parent` |
| john3 | `q2-directional-parent` |
| john4 | `q3-affordance-parent` |

Q4 is excluded because it failed ADR 0155's P99 token gate.

## Controlled Boundary

Only the once-per-decision parent token substrate differs. All arms share the
same:

- exact R2 active-board candidate afterstates;
- 560/240 open train/validation groups;
- train cohorts and all 860,203 validation actions;
- semantic supply and action facts;
- universal nine-class fixed-latent Perceiver model;
- 3,000-step optimizer and D6 schedule;
- objective and initialization; and
- validation and serving protocol.

## Predictions

1. Q1 should provide the largest parent-encode gain but has the greatest risk
   of losing directional frontier context.
2. Q2 should retain the strongest directional ranking quality.
3. Q3 should be strongest on habitat-growth and bridge decisions.
4. At least one bounded arm should match exact R2 aggregate quality.
5. A state-only parent replacement may improve parent latency without moving
   end-to-end candidate throughput enough to satisfy the material-efficiency
   rule; that is a valid null.

## Integrity

The Rust sidecar must align every group and public-state hash with the
accepted R3 cache, store all twelve exact D6 views, preserve every active
`i16` field, and prove no truncation. Python may only select and batch
Rust-authored views.

All arms must have identical parameter layouts, initial parameter tensors,
batch identities, candidate counts, and objective traces.

## Frozen Gates

Control sanity:

```text
MAE <= 1.42
RMSE <= 1.85
top-64 recall >= 0.70
top-64 regret <= 0.12
low-supply recall >= 0.88
independent-draft recall >= 0.76
confidence coverage >= 0.97
```

Treatment quality:

```text
MAE delta <= 0.05
RMSE delta <= 0.05
top-64 recall delta >= -0.005
top-64 regret delta <= 0.005
low-supply recall delta >= -0.01
independent-draft recall delta >= -0.01
confidence coverage >= 0.99
```

Material efficiency requires parent P50 at most 0.80x the same-host C0 plus at
least one of:

```text
throughput >= 1.05x same-host C0
complete P99 <= 0.95x same-host C0
peak active memory <= 0.85x same-host C0
peak RSS <= 0.85x same-host C0
```

## Claim Boundary

This experiment can select a bounded parent-state substrate. It cannot claim
pure bounded candidate encoding, gameplay improvement, or progress toward
100 points.
