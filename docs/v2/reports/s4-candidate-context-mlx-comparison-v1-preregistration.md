# S4 Candidate-Context MLX Comparison V1 Preregistration

Status: frozen before production optimizer execution

Date: 2026-06-17

ADR: 0153

Experiment: `s4-candidate-context-mlx-comparison-v1`

Protocol: `s4-candidate-context-matched-comparison-v1`

## Primary Question

Can exact whole-decision candidate context produce a material ranking gain on
the failed R3 radius-one substrate and recover the frozen full-R2 quality
envelope within a practical MLX serving bound?

## Binding Prior Result

The R3 classification must be exactly:

`r3_action_edit_mlx_all_treatments_degraded`

Bound identifiers:

```text
R3 classification ID =
  49260f87006bf9c49f145cd6de89db131ad916a9532d64b21c578201312404ae
R3 scientific BLAKE3 =
  4cc7d2deef805bbc3cc4584343f61aab2311253a5462b1aca56bfe6a70a19df9
R3 full-R2 report ID =
  75f2daf1ed8c70ac6fdeecb43f2a54efb0850c0969bc83a7f1ae0e08a569f562
R3 radius-one report ID =
  095ed5a6a4bd7ab096de61cd8bf80371edb06dad32dae417a8c8c81a8d87b418
```

The compact substrate is failed evidence, not a selected representation.

## Arms And Hosts

| Arm | Treatment | Host |
|---|---|---|
| `c0-independent` | Independent candidate scoring | `john1` |
| `t1-inducing-16` | Sixteen inducing latents | `john2` |
| `t2-exact-relations` | Six exact relation segments | `john3` |
| `t3-combined` | Both context mechanisms | `john4` |

No host runs a duplicate production arm.

## Frozen Inputs

Train dataset:

`artifacts/datasets/complete-action-graded-oracle-v1-train`

Validation dataset:

`artifacts/datasets/complete-action-graded-oracle-v1-validation`

R3 cache:

`0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156`

S1 exact-supply cache:

`2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15`

S4 context cache:

`fd3dcc8018cfe4b735a9a6514555e90e938fd142e746dc6d791f482e96463def`

Warm-start model:

`05befa5fae0c5f4af4f43f4b017ef74c1810b33f39b84bd8251c5f2a1a6e1919`

The sealed test split, gameplay results, hidden order, excluded identities, and
future refills are forbidden.

## Frozen Cohorts

| Split | Decisions | Candidate queries | Context anchors |
|---|---:|---:|---:|
| Train | 560 | 280,012 | At most 256 per decision |
| Validation | 240 | 860,203 | At most 256 per decision |

Train uses the exact R3 at-most-512 cohort. Validation scores every complete
action. Context anchors add evidence; they never remove queries.

## Initialization

Before step one:

```text
all four parameter counts equal
all four parameter-layout BLAKE3 values equal
all four initial-tensor BLAKE3 values equal
all four arms loaded from the same R3 radius-one checkpoint
context output deltas initialized to zero
S4 scores byte-identical to R3 radius-one scores
S4 standard errors byte-identical to R3 radius-one standard errors
```

Any failure invalidates launch.

## Frozen Training Protocol

```text
seed = 2026061721
optimizer = AdamW
training_steps = 3000
groups_per_step = 4
train_candidate_cap = 512
anchor_limit = 256
inducing_latents = 16
relation_neighbor_limit = 8
learning_rate = 3e-5
weight_decay = 1e-4
checkpoint_steps = 250
metric_steps = 250
validation_probe_groups = 12
candidate_chunk = 256
warm_start = true
warm_start_substrate = t3-r3-radius1-global
warm_start_substrate_status = failed-r3-compact-treatment
context_delta_zero_initialized = true
base_jointly_finetuned = true
early_stopping = false
```

The deterministic group and D6 schedules are identical across arms. Every
production report must contain exactly 3,000 ordered scientific batch hashes
and matching candidate counts.

## Frozen Loss

```text
r1200_huber
+ 4.0 * r4800_huber
+ 0.5 * r1200_listwise
+ 1.0 * r4800_winner
+ 0.1 * standard_error_calibration
+ 0.01 * screen_only_regularization
```

## Cross-Host Smoke

Before authorization, run `t3-combined` for exactly 10 steps on john1 and
john4 from the immutable bundle and identical inputs.

Exact gates:

```text
scientific batch hashes identical
candidate counts identical
initial tensor BLAKE3 identical
zero-context warm-start prediction identity identical
prediction-panel action hashes identical
stable prediction-panel ranking identical
```

Numerical tolerances:

```text
loss maximum absolute drift <= 1e-4
loss maximum relative drift <= 1e-5
checkpoint parameter maximum absolute drift <= 1e-4
checkpoint parameter mean absolute drift <= 1e-6
prediction score maximum absolute drift <= 1e-4
prediction uncertainty maximum absolute drift <= 1e-5
```

Production training remains false in both reports and the proof.

## All-Host Preflight

Every host independently verifies:

- immutable bundle content address;
- explicit authorization content address;
- exact train and validation manifests;
- R3, S1, and S4 context cache identities and semantics;
- final R3 failure classification;
- full-R2 and radius-one R3 report identities;
- failed radius-one checkpoint identity;
- source BLAKE3;
- Apple Silicon and MLX GPU;
- Python bytecode disabled;
- exact assigned host and arm;
- cross-arm initialization parity;
- zero-context prediction parity; and
- cross-host smoke proof.

All four preflights are dependencies of all four production arms.

## Complete Validation Evidence

Every arm must report:

```text
groups = 240
candidates = 860203
all groups scored once = true
all candidates scored once = true
finite scores and uncertainties = true
parent encodes = 240
parent encode count exact = true
prediction panel count = 64
```

Required slices:

- early;
- middle;
- late;
- low supply; and
- independent draft winner.

## Fresh Serving Evidence

The final checkpoint is loaded in a fresh process. Measurement excludes five
warmups and includes 30 steady iterations.

Fixed chunk:

```text
actions = 256
warmup iterations = 5
steady iterations = 30
```

Complete decisions:

```text
groups = 20
parent encodes = 20
anchor encodes = 20
both encode counts exact = true
```

The report binds request ID, result ID, checkpoint model BLAKE3, open-data
verification ID, context-cache ID, runtime, active memory, RSS, and swap.

## Absolute Gates

```text
complete validation coverage
all values finite
one parent and anchor encode per decision
process swap = 0
peak active memory <= 4 GiB
peak RSS <= 4 GiB
P99 complete-decision latency <= 250 ms
fixed-chunk throughput >= 20000 actions/s
```

## Quality Noninferiority To S4 Control

Every condition is required:

```text
MAE delta <= 0.05
RMSE delta <= 0.05
top-64 winner recall delta >= -0.005
top-64 retained-regret delta <= 0.005
low-supply recall delta >= -0.01
independent-winner recall delta >= -0.01
confidence coverage >= 0.99
```

## Material Context Effect

At least one condition is required:

```text
MAE reduction >= 0.05
RMSE reduction >= 0.05
top-64 recall gain >= 0.02
top-64 regret reduction >= 0.02
confidence coverage gain >= 0.01
```

## Full-R2 Quality Rescue

Every noninferiority condition is repeated against R3
`c0-full-r2-afterstate`.

The observed numerical envelope is:

```text
MAE <= 1.3702285
RMSE <= 1.7923126
top-64 recall >= 0.7200000
top-64 retained regret <= 0.10311535
low-supply recall >= 0.9022807
independent-winner recall >= 0.7995238
confidence coverage >= 0.99
```

## Full-R2 Serving Rescue

Every condition is required:

```text
fixed throughput >= 0.25x R3 full-R2
complete-decision P99 <= 1.75x R3 full-R2
peak active memory <= 2.50x R3 full-R2
peak RSS <= 1.50x R3 full-R2
```

These relative bounds coexist with the absolute gates.

## Frozen Classification

Precedence:

1. `s4_candidate_context_mlx_invalid_evidence`
2. `s4_candidate_context_mlx_control_failed`
3. `s4_candidate_context_mlx_all_treatments_degraded`
4. `s4_candidate_context_mlx_context_null`
5. `s4_candidate_context_mlx_context_signal_only`
6. `s4_candidate_context_mlx_compact_rescue_selected`

Interpretation:

- `all_treatments_degraded`: no context arm is quality-noninferior to the S4
  independent control.
- `context_null`: at least one treatment is noninferior, but none has a
  material context effect.
- `context_signal_only`: context materially helps versus the matched compact
  control, but no arm recovers the full-R2 quality and serving envelope.
- `compact_rescue_selected`: one or more treatments pass all matched-control,
  material-effect, full-R2 quality, and full-R2 serving gates.

Forward and reverse report orders must produce byte-identical classification
files.

## Promotion Boundary

No S4 result directly authorizes model promotion. Only
`compact_rescue_selected` authorizes a subsequent paired-gameplay gate.

## Nonclaims

This experiment does not measure:

- benchmark mean game score;
- paired gameplay;
- progress above 100;
- search quality;
- sealed-test performance; or
- final model promotion.
