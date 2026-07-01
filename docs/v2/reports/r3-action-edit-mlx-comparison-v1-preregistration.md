# R3 Action-Edit MLX Comparison V1 Preregistration

Status: frozen before cache export or optimizer execution

Date: 2026-06-17

ADR: 0150

Experiment: `r3-action-edit-mlx-comparison-v1`

Protocol: `r3-action-edit-mlx-matched-comparison-v1`

## Primary Question

Can exact R3 canonical local-patch plus global-edit tokens match the held-out
complete-action ranking quality of a full exact R2 afterstate while materially
improving MLX throughput, latency, or memory?

## Arms And Hosts

| Arm | Representation | Host |
|---|---|---|
| `c0-full-r2-afterstate` | Complete exact canonical R2 active-board afterstate | `john1` |
| `t1-r3-radius3-global` | Radius 3 plus exact global edits | `john2` |
| `t2-r3-radius2-global` | Radius 2 plus exact global edits | `john3` |
| `t3-r3-radius1-global` | Radius 1 plus exact global edits | `john4` |

No host runs a duplicate primary arm. Cross-host replay is a bounded
pre-production smoke, not a fifth treatment.

## Frozen Open Data

| Split | Games | Decisions | Complete actions | Use |
|---|---:|---:|---:|---|
| Train | 7 | 560 | 2,135,111 | Deterministic at-most-512 cohort per decision |
| Validation | 3 | 240 | 860,203 | Every action scored exactly once |

Train root:

`artifacts/datasets/complete-action-graded-oracle-v1-train`

Validation root:

`artifacts/datasets/complete-action-graded-oracle-v1-validation`

Exact semantic-supply sidecar:

`artifacts/experiments/exact-semantic-supply-learned-comparison-v1/cache/2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15`

The sealed test split and gameplay artifacts are forbidden.

## Train Candidate Selection

The exporter implements the exact ADR 0150 seven-stage sampler. It records,
for every decision:

- source action count;
- retained count;
- every retained source index and action hash;
- selected and champion retention;
- R600 and R4800 retention;
- screen-rank histogram; and
- cohort identity BLAKE3.

Production requirements:

```text
selected winner retained == 560 / 560
champion retained == 560 / 560
R600 actions retained == all
R4800 actions retained == all
groups wider than 512 retain exactly 512
groups at most 512 retain every action
duplicate retained indices == 0
```

## Mechanical Export Gates

For every cached group:

```text
source PositionRecord parity == exact
public-state hash parity == exact
public-supply parity == exact
parent R2 encoding count == 1
```

For every cached candidate:

```text
graded action reconstruction == exact
grouped R3 observation contains action == exact
R3 apply parity == exact
authoritative public successor parity == exact
canonical transform reproduction == exact
full R2 afterstate token multiset == exact
control parent-delta reconstruction == exact
R3 MLX token encode/decode round trip == exact
action-hash alignment == exact
S1 candidate alignment == exact
silent truncation, clipping, or overflow == 0
```

The cache is production-eligible only if it covers all 800 open decisions,
the frozen train cohort, and all 860,203 validation actions.

## Frozen Model And Training

```text
seed = 2026061708
optimizer = AdamW
steps = 3000
groups_per_step = 4
candidate_cap_per_train_group = 512
learning_rate = 0.0001
weight_decay = 0.0001
checkpoint_interval = 250
metric_interval = 100
full_validation_runs = 1
candidate_chunk = 256
hidden_dim = 64
attention_heads = 4
parent_perceiver_latents = 16
candidate_perceiver_latents = 8
parent_latent_blocks = 1
candidate_latent_blocks = 1
warm_start = false
early_stopping = false
```

Three batch slots use independent deterministic permutations of all train
groups. The fourth alternates between low-supply and independent-winner
permutations. All four arms consume the identical ordered group, candidate,
target, and D6 stream.

Parameter count, parameter layout BLAKE3, and initial tensor BLAKE3 must be
identical across all arms before step one.

## Frozen Loss

```text
r1200_huber
+ 4.0 * r4800_huber
+ 0.5 * r1200_listwise
+ 1.0 * r4800_winner
+ 0.1 * standard_error_calibration
+ 0.01 * screen_only_regularization
```

No arm-specific auxiliary term is permitted.

## Validation Evidence

Every report must contain:

- exact decision and action coverage;
- R4800 MAE, RMSE, bias, correlation, slope, and intercept;
- top-1/8/32/64 stable-winner recall;
- top-1/8/32/64 retained regret;
- top-64 95% confidence-set coverage;
- early/middle/late metrics;
- low-supply metrics;
- independent-draft-winner metrics;
- fixed action-hash prediction panel;
- parent encode count;
- action-token count and padding distributions;
- action scores per second;
- P50/P95/P99 complete-decision latency;
- compile, warmup, and steady timings;
- MLX active/cache/peak memory;
- process peak RSS and swap;
- system swap before/after;
- final checkpoint identity; and
- complete information-boundary assertions.

Serving performance and process RSS must come from a fresh worker that reloads
the report's exact final checkpoint. The worker request and result, checkpoint
model BLAKE3, open-data verification identity, and runtime identity are part of
the arm report. Exhaustive sidecar and source-action verification remains a
mandatory preflight process and is deliberately excluded from serving RSS.
The frozen rationale and smoke evidence are in
`docs/v2/reports/r3-action-edit-mlx-serving-rss-amendment-2026-06-17.md`.

## Frozen Classification

Classification precedence:

1. `r3_action_edit_mlx_invalid_evidence`
2. `r3_action_edit_mlx_control_failed`
3. `r3_action_edit_mlx_all_treatments_degraded`
4. `r3_action_edit_mlx_quality_only_null`
5. `r3_action_edit_mlx_compact_representation_selected`

`quality_only_null` means at least one R3 arm matched control quality but none
met the material efficiency gate.

`all_treatments_degraded` means no R3 arm met every quality-noninferiority
gate.

`compact_representation_selected` names exactly one selected arm by the ADR
0150 deterministic ordering.

Forward and reverse report orders must produce byte-identical classification
files. A separate order proof binds both bytes and the scientific
classification BLAKE3.

## Pre-Production Numerical-Parity Amendment

The first bounded john1/john4 smoke established that MLX GPU reductions are
not bitwise deterministic. Repeating the same 10-step radius-1 smoke on john1
also changed low-order floating-point bits, so byte-identical loss,
checkpoint, and prediction tensors are not a valid cross-host requirement.

Production had not started when this was discovered. The exact requirements
remain:

- ordered scientific batch BLAKE3 values;
- candidate counts per step;
- initial parameter tensor BLAKE3;
- parameter count and layout BLAKE3;
- prediction-panel action hashes; and
- stable prediction-panel ranking.

The floating-point parity gates are frozen at:

```text
loss max absolute drift <= 1e-4
loss max relative drift <= 1e-5
checkpoint parameter max absolute drift <= 1e-4
checkpoint parameter mean absolute drift <= 1e-6
prediction score max absolute drift <= 1e-4
prediction uncertainty max absolute drift <= 1e-5
```

The comparator must checksum both complete checkpoints against their reports,
load every parameter tensor, reject layout or finite-value drift, and emit a
content-addressed proof. The empirical basis and first passing proof are in
`docs/v2/reports/r3-action-edit-mlx-cross-host-smoke-amendment-2026-06-17.md`.

## Nonclaims

This experiment does not measure:

- paired gameplay;
- benchmark mean score;
- progress above 100 mean;
- search quality;
- opponent modeling;
- model promotion; or
- sealed-test generalization.

A passing result authorizes the next paired-gameplay gate only.
