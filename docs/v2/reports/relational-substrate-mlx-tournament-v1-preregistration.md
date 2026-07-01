# Relational Substrate MLX Tournament V1 Preregistration

Date: 2026-06-17

ADR: 0161

Experiment: `relational-substrate-mlx-tournament-v1`

Protocol: `r5-s3-s5-matched-mlx-v1`

Status: frozen before production cache export or optimizer execution

## Primary Question

Can exact component, motif, topology, and counterfactual structure recover or
improve the accepted exact-R2 complete-action ranking quality while making
the full candidate decision materially faster?

## Arms And Hosts

| Arm | Representation | Host |
|---|---|---|
| `c0-exact-r2` | exact R2 parent and exact R2 candidate afterstate | `john1` |
| `q1-r5-quotient-local` | R5-minimal quotient and exact radius-one R3 action edit | `john2` |
| `g2-r5-s3` | rich S3 relational parent and exact radius-one R3 action edit | `john3` |
| `d3-r5-s3-s5` | rich S3 parent, exact radius-one R3 action edit, and S5 derivative | `john4` |

No host runs a duplicate primary arm. Same-host exact-control replays are
serving controls only.

## Immutable Inputs

```text
R3 cache:
  0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156
S1 cache:
  2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15
train decisions:
  560
train source actions:
  2,135,111
train retained actions:
  280,012
validation decisions:
  240
validation retained actions:
  860,203
```

The sealed test split, gameplay artifacts, hidden order, excluded tile
identity, and future refill are forbidden.

## Sidecar Export Gates

For every group:

```text
group ID == R3
public-state hash == R3
source candidate count == R3
retained candidate offsets and source indices == R3
candidate identity hash == R3
public PositionRecord replay == exact
public supply == exact
12 D6 rich-parent views exported
R5 current-score decoder == exact
opportunity-family flags derived without targets
```

For every retained candidate:

```text
source action hash == R3
R3 grouped action reconstruction == exact
R3 apply afterstate hash == exact
S5 immediate score delta == R3
S5 feature width == 154
all raw S5 values fit signed 16-bit storage
silent clipping or truncation == 0
```

The sidecar is complete only at 800 groups, 9,600 parent D6 views, and
1,140,215 retained candidates.

## Frozen Model

```text
hidden_dim = 64
attention_heads = 4
parent_perceiver_latents = 16
candidate_perceiver_latents = 8
parent_latent_blocks = 1
candidate_latent_blocks = 1
cross_board_blocks = 1
staged_market_blocks = 1
relational_value_width = 64
relational_classes = 8
s5_derivative_width = 154
feed_forward_multiplier = 2
```

The graph includes native R2 and relational token adapters in every arm.
Masked or zero factual surfaces remain present in the parameter graph.

Before step one:

```text
parameter count identical across arms
parameter layout BLAKE3 identical across arms
initial parameter tensor BLAKE3 identical across arms
first bounded batch identity identical across arms
initial prediction ranking identical when factual inputs are forced equal
```

## Frozen Training

```text
seed = 2026061716
optimizer = AdamW
steps = 3000
groups_per_step = 4
train candidate cap = 512
learning_rate = 0.0001
weight_decay = 0.0001
checkpoint interval = 250
metric interval = 100
candidate chunk = 256
warm start = false
early stopping = false
```

The ordered group, source candidate, action hash, target, and D6 streams are
identical. The loss is unchanged from ADR 0150:

```text
r1200_huber
+ 4.0 * r4800_huber
+ 0.5 * r1200_listwise
+ 1.0 * r4800_winner
+ 0.1 * standard_error_calibration
+ 0.01 * screen_only_regularization
```

No representation-specific auxiliary loss is permitted.

## Strategic Subsets

The exact parent graph marks whether the active board has:

- an Elk eligible extension;
- a Salmon legal continuation;
- a Hawk isolated-placement opportunity; and
- a Bear pair-completion opportunity.

The primary strategic metric is the unweighted mean top-64 winner recall over
the Elk, Salmon, and Hawk subsets. Bear remains diagnostic because the
historical player already over-indexed on Bear.

## Frozen Quality Gates

Control:

```text
MAE <= 1.42
RMSE <= 1.85
top-64 recall >= 0.70
top-64 regret <= 0.12
low-supply recall >= 0.88
independent-draft recall >= 0.76
confidence coverage >= 0.97
```

Treatment relative to control:

```text
MAE <= control + 0.05
RMSE <= control + 0.05
top-64 recall >= control - 0.005
top-64 regret <= control + 0.005
low-supply recall >= control - 0.01
independent-draft recall >= control - 0.01
confidence coverage >= 0.99
strategic recall mean >= control + 0.015
each Elk/Salmon/Hawk recall >= control - 0.01
```

## Frozen Serving Gates

Every report contains model-only, materialization, R6, and combined timings.

Absolute:

```text
model-only fixed-chunk throughput >= 20,000 actions/s
combined complete-decision P99 <= 250 ms
peak MLX active memory <= 4 GiB
peak process RSS <= 4 GiB
process swaps == 0
R6 apply/undo failures == 0
```

Material efficiency against a same-host exact-control replay:

```text
combined actions/s >= 1.10 * paired control
OR
combined P99 <= 0.90 * paired control
```

Cache construction and exhaustive source verification are preflight work and
are excluded from serving RSS. Per-decision feature materialization is
included.

## Frozen Classification

```text
relational_substrate_mlx_invalid
relational_substrate_mlx_control_failed
relational_substrate_mlx_all_treatments_degraded
relational_substrate_mlx_quality_only_null
relational_substrate_mlx_selected
```

Forward and reverse report order must produce byte-identical scientific
classification.

## Predictions

1. Q1 will be materially faster than exact R2 but may remain value-inferior.
2. G2 will recover some of the R3 value gap, especially on Elk, Salmon, and
   Hawk opportunity subsets.
3. D3 will most strongly affect low-supply and independent-draft decisions.
4. A useful compact substrate will need both explicit long-range state and
   exact local action evidence; local radius alone will not be sufficient.

## Invalidators

- any production optimizer step before immutable authorization;
- changed candidate cohort, labels, D6 schedule, or train order;
- validation-fitted S5 normalization;
- arm-specific parameter count or initialization;
- missing R6 parity;
- post-hoc quality or efficiency thresholds;
- hidden or sealed data access;
- duplicate primary work across hosts; or
- scientific classification that depends on report order.

## Claim Boundary

A selected arm authorizes paired gameplay qualification. This tournament does
not itself alter the champion or establish progress above a 100-point mean.
