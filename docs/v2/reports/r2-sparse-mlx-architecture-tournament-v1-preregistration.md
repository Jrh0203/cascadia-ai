# R2 Sparse MLX Architecture Tournament V1 Preregistration

Date: 2026-06-17

ADR: 0146

Experiment: `r2-sparse-mlx-architecture-tournament-v1`

Protocol: `r2-sparse-mlx-matched-architecture-v1`

Status: completed; Perceiver fixed latents selected

## Research Question

Given the identical exact ADR 0145 R2 public tokens, targets, D6 schedule,
training rows, optimizer steps, seed, and approximately matched full model
capacity, which local MLX trunk gives the best validation value quality and
serving performance?

The screen compares:

1. padded Set Transformer;
2. directional graph message passing plus global attention; and
3. Perceiver-style fixed latents.

An independent Set Transformer replay on john4 tests reproducibility.

## Frozen Evidence Domain

Only the eight accepted ADR 0145 train and validation roots are admissible:

```text
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-0
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-1
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-2
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-3
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-0
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-1
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-2
artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-3
```

Frozen totals and identities:

| Field | Value |
|---|---|
| Train rows | 50,000 |
| Validation rows | 10,000 |
| Total rows | 60,000 |
| Feature schema | `compact-entity-v2` |
| Target schema | `base-score-components-v1` |
| Strategy | `pattern-aware-v1-k8-h6-b8-m4` |
| Players | 4 |
| Wildlife cards | AAAAA |
| Habitat bonuses | false |
| Foundation scientific BLAKE3 | `186ad8934287ef0a74a166ed00cc9ebe857dcded20faa01a264974e1eb7081e6` |
| Foundation public-position BLAKE3 | `29836be57c6e0529c06b0b628c455b27f06284fe7a8c333e54024174a7e7f003` |
| Foundation packed-state BLAKE3 | `c181be2126a42b668f500666cccf41573ea079a3f2c34ab7bc3989f690fec789` |

Test and final splits are prohibited.

## Exact Input Contract

All runs consume one content-addressed cache with:

- four explicit relative-board blocks;
- padded capacity 92 tokens per board and 368 per position;
- no clipping, pooling, overflow loss, or truncation;
- explicit four-way relative-seat ownership on every active token;
- 52 exact payload fields per token;
- exact Rust-authored graph CSR;
- 23 public features for each of four player slots;
- 31 market features;
- 96 public global features; and
- the unchanged 11 base score-component targets.

The foundation maximum is 340 active tokens across four boards and 92 on one
board; per-board P99 is 83. The frozen full-corpus type totals are:

| Token class | Total | Mean per position |
|---|---:|---:|
| Occupied | 3,090,000 | 51.5000 |
| Legal frontier | 4,155,914 | 69.2652 |
| Habitat component | 2,257,600 | 37.6267 |
| Wildlife motif | 2,365,940 | 39.4323 |
| **Total** | **11,869,454** | **197.8242** |

Every report must include padded capacity and split-specific active-token
total, mean, maximum, maximum per board, padding total, and per-type totals,
means, fractions, and maxima.

`CSR2SP1` contains authoritative occupied/public state, not redundant derived
frontier, component, motif, or graph state. Rust must regenerate those
projections, validate the frozen census, prove all 12 D6 transform/inverse
pairs for every source row, and freeze the validated derived tensors in the
content-addressed cache. Python may apply only the frozen accepted D6 tables
to cached payload fields and direction bits. It must not reconstruct
frontiers, components, motifs, neighborhoods, or relation types.

Within each board, active rows are contiguous in canonical type order:
occupied, frontier, habitat component, wildlife motif. Raw token
attention/message passing is board-local. Cross-board interaction is allowed
only through explicit global, market, and player-board summary tokens.

## Arms And Hosts

| Run role | Architecture | Host | Role |
|---|---|---|---|
| `set-primary` | padded Set Transformer | john1 | primary |
| `graph-primary` | directional graph attention | john2 | primary |
| `perceiver-primary` | Perceiver fixed latents | john3 | primary |
| `set-replay` | padded Set Transformer | john4 | independent replay |

The three primary runs become ready together after one cache fanout. The
replay has the same priority and budget because it can invalidate the entire
screen if the protocol is not reproducible.

## Full Parameter Gate

Counts include every trainable adapter, embedding, common encoder weight,
architecture trunk weight, summary projection, and output head.

| Architecture | Parameters |
|---|---:|
| Padded Set Transformer | 141,131 |
| Directional graph attention | 143,915 |
| Perceiver fixed latents | 142,283 |

Required spread:

```text
(max_parameter_count - min_parameter_count) / min_parameter_count <= 0.03
```

Observed implementation spread is 0.0197264. Unused branch parameters and
nontraining ballast are forbidden.

## One-Encoding Invariant

For one prediction:

```text
common_state_encoder_invocations == 1
```

Every architecture must pass a focused instrumented test. Masked padding may
contain arbitrary values without changing predictions within `rtol=1e-5` and
`atol=1e-5`.

Final board pooling must use one player token plus four equal-status
type-summary tokens, not a raw count-weighted mean over all active rows. The
training report must include four validation ablations:

```text
occupied masked
frontier masked
habitat component masked
wildlife motif masked
```

Ablation masks the frozen token rows and their incident cached graph edges. It
does not regenerate a counterfactual semantic state.

## Frozen Training Protocol

```text
seed: 2026061702
optimizer: AdamW
steps: 500
batch size: 32
learning rate: 0.0003
weight decay: 0.0001
checkpoint interval: 100
metric interval: 25
evaluation batch: 64
inference batch: 64
D6: uniform per example over IDs 0..11
```

Training row and transform IDs are derived solely from the seed and optimizer
step. Every run therefore sees the same 16,000 optimizer examples.

## Required Measurements

### Learning

- full train and validation loss;
- per-component MAE, RMSE, and bias;
- mean component MAE;
- total MAE, RMSE, and bias;
- total correlation;
- calibration slope and intercept;
- per-type validation masking-ablation deltas; and
- fixed first-256 validation predictions under identity D6.

### Performance

- first compiled invocation;
- warm and steady examples per second;
- inference actions per second;
- P50, P90, and P99 latency;
- forward-plus-backward examples per second;
- MLX active, cache, and peak memory;
- cumulative training examples per second; and
- process peak RSS.

Cross-host throughput is operational evidence and is not used as a direct
architecture-quality ratio.

## Independent Replay Gate

`set-primary` and `set-replay` must satisfy:

```text
abs(validation_total_mae_delta) <= 0.10
abs(validation_total_rmse_delta) <= 0.15
abs(validation_mean_component_mae_delta) <= 0.03
max_abs(first_256_component_prediction_delta) <= 0.10
```

Replay failure blocks architecture selection.

## R0 Reference Gate

The R0 reference is selected fail-closed:

```text
if r0.selected_stage2_candidate is not null:
    reference = selected_stage2_candidate
else:
    reference = exact-entity-control
```

The null branch is legal only when the completed R0 classifier explicitly
contains null and forward/reverse classification bytes match.

Each R2 primary is R0-value-noninferior when:

```text
r2.validation.total_mae - r0.validation.total_mae <= 1.0
r2.validation.total_rmse - r0.validation.total_rmse <= 1.5
r2.validation.mean_component_mae
    - r0.validation.mean_component_mae <= 0.25
```

## Selection

Only R0-value-noninferior primary runs are eligible. After replay passes,
selection order is:

1. lower validation total MAE;
2. lower validation mean component MAE;
3. lower inference P50 latency; and
4. canonical architecture ID.

No eligible primary is a valid completed result with no selected architecture.
It does not authorize promotion.

## Classification

Fail-closed precedence:

1. semantic failure;
2. structural incompleteness;
3. insufficient performance evidence;
4. independent replay failure;
5. complete.

The classifier runs on forward and reverse report orders. Output bytes must be
identical.

## Immutable Launch Gates

Before cache export:

1. build the standalone release exporter;
2. create and validate an immutable source/binary bundle;
3. freeze the corpus lock;
4. bind the completed R0 selection and order proof;
5. create explicit parent authorization; and
6. pass Apple Silicon MLX GPU preflight on john1 through john4.

Before any optimizer step:

1. export the exact cache once on john1;
2. validate its content address, complete tensor file set, checksums, exact
   type census, board-local padding, cached relations, and semantic proofs;
3. fan out the complete cache tree byte-identically; and
4. validate the cache binding on the assigned host.

The campaign's `queue-spec` output is inert. It has no `--apply` option, does
not import the live queue library, and cannot modify the dashboard ledger.

## Authorized Production Identity

The production launch was authorized on 2026-06-17 for exactly the four
preregistered run roles. No additional architecture, seed, corpus, target,
optimizer step, or host substitution is authorized.

```text
bundle:
  65613a7526f292af47740c1ef21bc3bca22efa5454f0a9b3948c037f2db2a962
authorization:
  780455bfd22643c07ec8c9624d4e2c81342f7f3887a49087b7ea1bc759e5b6a3
corpus lock:
  264315f193ea69232c699277f503c50aa1e1026eb1d27fecd7127503ff6ae0f7
R0 control binding:
  a314077de5cce28dabd7d25d17bd6520fb830c85d2c6f624688aa7875615e8a6
exporter BLAKE3:
  a93b723af21c241da8d5813a87fa42004672d96b199b40c14622c21e06c5a974
MLX source BLAKE3:
  dfb035ef7b93ebbd5d230cb80e743894aa493cd90e1879ce4418086e26ba4173
queue task graph BLAKE3:
  d40000a0c0b790668421a5d5fa001ab5c9bc3359ffb9771b6380e4f1b6e2a11e
```

The live scheduler installation must contain all sixteen preregistered tasks
or none of them. Scientific execution remains dependent on all four host
preflights and the content-addressed cache fanout.

## Completed Result

All sixteen production tasks completed on 2026-06-17. The independent replay
passed exactly, forward/reverse classifications were byte-identical, and the
classifier selected `perceiver-fixed-latents` as the only primary satisfying
all frozen R0 value-noninferiority gates.

The result does not authorize promotion or make a gameplay claim. See
`docs/v2/reports/r2-sparse-mlx-architecture-tournament-v1-result.md`.

## Preparation Commands

These commands define the launch package but are not authorization to run it:

```bash
cargo build --release \
  --manifest-path tools/r2_sparse_entity_census/Cargo.toml

PYTHONPATH=python:tools .venv/bin/python -B \
  tools/r2_sparse_mlx_campaign.py freeze-corpus

PYTHONPATH=python:tools .venv/bin/python -B \
  tools/r2_sparse_mlx_campaign.py bind-r0-control
```

Build the immutable bundle with `tools/rust_experiment_bundle.py`, including
the root manifests, `crates/cascadia-data`, `crates/cascadia-game`,
`python/cascadia_mlx`, `tools/r2_sparse_entity_census`, the two R2 tools,
`tools/cluster_artifact_fanout.py`, and the release exporter binary.

Authorization requires a nonempty approver and is intentionally a separate
command. No authorization artifact, production cache, production checkpoint,
live queue task, or dashboard ledger entry is created by implementation
verification.

## Implementation Verification

Allowed before launch:

- focused Python unit tests;
- synthetic MLX forward/backward and compile smokes;
- standalone Rust unit/integration tests;
- Ruff;
- Rustfmt; and
- Clippy.

Prohibited before explicit launch:

- the 60,000-row MLX cache export;
- any 500-step production architecture run;
- live queue installation;
- dashboard ledger modification; and
- gameplay or promotion claims.
