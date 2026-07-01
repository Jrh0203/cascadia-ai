# ADR 0142: R0 Spatial MLX Iso-Architecture Tournament

Status: accepted and completed

Date: 2026-06-17

Experiment: `r0-spatial-mlx-tournament-v1`

Protocol: `r0-spatial-mlx-iso-architecture-v1`

## Context

ADR 0135 established five lossless Rust spatial representations over the
canonical `compact-entity-v2` public position:

1. exact entities;
2. recentered radius 6;
3. recentered radius 5;
4. recentered radius 4; and
5. the historical fixed-origin 21 by 21 axial square.

ADRs 0136 and 0138-0141 established the extraction classifier, the exact
60,000-row production corpus, work-conserving scheduling, and uncontended
timing recovery. That evidence measures Rust extraction, packed bytes, and
round trips. It does not answer whether the representations remain sufficient
after learning or how their tensor shapes behave under MLX.

R0 Stage 2 must change only the spatial representation. It must not obtain a
speedup by changing targets, dropping overflow entities, using a smaller
network, using fewer optimizer steps, opening test data, or training on a
different random stream.

The cluster contains four Apple Silicon Macs. Four decision-changing arms can
run concurrently. The fifth arm, historical 441, is diagnostic and must
backfill the first host that finishes a primary arm rather than delaying a
compact arm.

## Decision

The R0 Stage 2 implementation consists only of new, experiment-specific
surfaces:

- `crates/cascadia-data/src/bin/r0_spatial_mlx_export.rs`;
- `python/cascadia_mlx/r0_spatial_mlx_cache.py`;
- `python/cascadia_mlx/r0_spatial_mlx_model.py`;
- `python/cascadia_mlx/r0_spatial_mlx_tournament.py`;
- `tools/r0_spatial_mlx_campaign.py`;
- `tools/r0_spatial_mlx_report.py`;
- focused tests; and
- this ADR and its preregistration.

The existing spatial contract, data library, legacy NNUE implementation,
scheduler core, dashboard, and R0 ledger are not changed.

## Frozen Corpus

`freeze-corpus` accepts exactly the eight ADR 0138 dataset roots in canonical
order:

| Order | Split | Part | First game | Games | Rows |
|---:|---|---:|---:|---:|---:|
| 0 | train | 0 | 200000 | 157 | 12560 |
| 1 | train | 1 | 200157 | 156 | 12480 |
| 2 | train | 2 | 200313 | 156 | 12480 |
| 3 | train | 3 | 200469 | 156 | 12480 |
| 4 | validation | 0 | 210000 | 32 | 2560 |
| 5 | validation | 1 | 210032 | 31 | 2480 |
| 6 | validation | 2 | 210063 | 31 | 2480 |
| 7 | validation | 3 | 210094 | 31 | 2480 |

Every manifest and physical shard is checksummed. The tool requires:

- schema version 1;
- `compact-entity-v2`;
- `base-score-components-v1`;
- four-player Standard mode;
- A scoring for all five wildlife;
- habitat bonuses disabled;
- `pattern-aware-v1-k8-h6-b8-m4`;
- complete contiguous game intervals;
- 80 positions per game;
- 50,000 train rows and 10,000 validation rows; and
- one shared V2 source digest.

The corpus lock records each raw manifest BLAKE3. Its corpus digest is:

```text
BLAKE3(
    "R0MLXCORPUS1\0"
    || little_endian_u64(len(manifest_0)) || manifest_0
    || ...
    || little_endian_u64(len(manifest_7)) || manifest_7
)
```

The exporter recomputes the same sequence digest and revalidates every dataset
with `cascadia-data` before opening records. Test and final splits are rejected.

## Rust Semantic Authority

The exporter calls `SpatialPositionRepresentation` and `D6Transform` directly.
Python does not reproduce:

- minimax recentering;
- local index order;
- exact overflow classification;
- tile-orientation transforms;
- packed round trips; or
- inverse D6 behavior.

For every one of the 60,000 source records and every arm, Rust proves:

```text
decode(encode(record)) == record
decode(pack(encode(record))) == record
inverse(transform(encode(record))) == record semantics
```

All 12 rules-owned D6 transforms are exported. The cache records separate
BLAKE3 digests for source records, transformed records, and targets. These
digests must be identical across arms.

## MLX Cache Contract

Each board has at most 23 occupied entities. The on-disk cache stores:

- Rust-selected dense destination slots;
- one sparse row per occupied entity;
- all-zero padding rows;
- decoded market features;
- decoded global features;
- unchanged 11-component targets;
- game index;
- turn;
- board counts; and
- every D6 transform.

Python performs only a generic scatter from the Rust destination slot into the
declared dense tensor. It rejects:

- a cache directory whose name is not its content address;
- corpus-lock drift;
- tensor path escape;
- shape, dtype, byte-count, or checksum drift;
- nonzero padding;
- duplicate or out-of-range slots;
- path-code and destination disagreement;
- active-row and board-count disagreement;
- overflow accounting drift; and
- missing Rust round-trip proofs.

The explicit per-board MLX shapes are:

| Arm | Local rows | Exact overflow reserve | Tensor rows |
|---|---:|---:|---:|
| `exact-entity-control` | 0 | 23 exact rows | 23 |
| `hex-radius-6-127` | 127 | 23 | 150 |
| `hex-radius-5-91` | 91 | 23 | 114 |
| `hex-radius-4-61` | 61 | 23 | 84 |
| `historical-square-21x21-441` | 441 | 23 | 464 |

The overflow reserve is not a pooled summary. Every out-of-region entity
remains exact and individually addressable. Unused local or overflow rows are
masked and all zero.

Caches are installed at:

```text
OUTPUT_ROOT/<cache_id>/cache.json
```

`cache_id` is the canonical JSON BLAKE3 of the scientific identity, including
the corpus lock, arm, tensor files, semantic digests, target digest, D6 IDs,
split counts, exporter executable BLAKE3, and exporter V2 source digest.

## Frozen Model

Every arm uses one parameterization:

```text
r0-spatial-iso-set-value-v1
hidden width: 32
attention heads: 4
board attention blocks: 1
feed-forward multiplier: 2
trainable parameters: 74,635
outputs: 11 nonnegative score components
```

The token encoder contains:

- a projection of absolute and carried-center coordinates;
- path, terrain, rotation, wildlife-mask, placed-wildlife, and keystone
  embeddings;
- a relative-seat embedding;
- one masked self-attention block per board;
- masked mean and max pooling;
- the shared 31-wide market decoder;
- the shared 96-wide global decoder; and
- one shared value trunk.

Parameter count is independent of sequence length. Padding is zeroed before
attention, excluded as an attention key, zeroed after every block, and
excluded from pooling. Tests require identical predictions for one active
position represented in 23 rows or zero-padded to 464 rows.

The loss combines normalized component MSE with total-score MSE:

```text
loss = mean(component_error_normalized^2)
     + 0.5 * mean((predicted_total - target_total)^2 / 100^2)
```

The model reports component MAE, RMSE, and bias, plus total MAE, RMSE,
correlation, bias, calibration slope, and calibration intercept.

## Frozen Optimization Protocol

The protocol is content-hashed into the production authorization:

| Variable | Value |
|---|---:|
| Seed | 2026061701 |
| Optimizer | AdamW |
| Steps | 500 |
| Batch size | 32 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Checkpoint interval | 100 steps |
| Metric interval | 25 steps |
| Evaluation batch | 64 |
| Inference batch | 64 |
| Inference warmup | 5 invocations |
| Inference steady sample | 30 invocations |
| Gradient warmup | 2 invocations |
| Gradient steady sample | 10 invocations |
| D6 policy | Uniform per example over Rust IDs 0 through 11 |
| MLX cache limit | 1 GiB |

Training batches are a deterministic function of `(seed, optimizer_step)`.
Resume therefore reconstructs the same sample and D6 transform without storing
an implicit RNG cursor. Resume validation rejects source, runtime, corpus,
cache, protocol, or authorization drift.

## Performance Measurements

Each run reports:

- first compiled invocation, including mandatory execution;
- warmup examples per second;
- steady-state examples per second;
- inference actions per second, where one scored afterstate is one action;
- P50, P90, and P99 latency;
- MLX active, cache, and peak memory;
- process peak RSS;
- cumulative optimizer examples per second;
- same-host forward-plus-backward examples per second; and
- train and full validation metrics.

Raw throughput across unlike Macs is operational evidence only. Every arm also
runs the trained model against a 23-token exact-shape packing on the same host.
It measures both inference and forward-plus-backward throughput without
changing parameters. The classifier uses these same-host ratios for leverage
gates, avoiding a false speedup caused by host hardware.

## Production Authorization

No production optimizer step can begin without a parent-created authorization
that pins:

- ADR 0142 and the experiment ID;
- the complete protocol hash;
- corpus-lock ID;
- complete MLX source digest;
- immutable bundle ID;
- exporter executable BLAKE3;
- all five authorized arm IDs;
- approver; and
- approval timestamp.

The tournament runner independently verifies the protocol, source, corpus,
arm, and exporter checksum against this authorization.

The host preflight additionally requires:

- the immutable bundle source directory;
- a valid bundle manifest;
- all eight local dataset trees matching the corpus lock;
- Apple Silicon macOS;
- the MLX GPU device;
- matching Python and MLX runtime identity; and
- the frozen 74,635 parameter model.

Preflight explicitly records that training has not started.

## Work-Conserving Queue

`queue-spec` creates 15 tasks:

1. immutable bundle fanout;
2. corpus-lock and authorization fanout;
3. four host-local preflights;
4. exact control on john1;
5. radius 6 on john2;
6. radius 5 on john3;
7. radius 4 on john4;
8. historical 441 on any of john1-4;
9. dynamic report collection;
10. forward classification;
11. reverse classification; and
12. byte-level classification-order proof.

The four primary arms have priority 10 and one compatible host each.
Historical 441 has priority 20, is compatible with all four hosts, and depends
on all preflights. Once the primary arms are claimed, the first host to finish
can claim the diagnostic arm. It cannot displace a decision-changing arm.

Report collection reads the completed queue result for every arm and retrieves
the artifact from the recorded host. Historical host assignment therefore
remains dynamic but auditable.

The generated queue specification is inert unless `--apply` is supplied.
Building this implementation does not mutate the queue or launch training.

## Stage 2 Classifier

Classification precedence is:

1. `r0_spatial_mlx_tournament_semantic_failure`;
2. `r0_spatial_mlx_tournament_incomplete`;
3. `r0_spatial_mlx_tournament_insufficient_performance_evidence`; and
4. `r0_spatial_mlx_tournament_complete`.

A complete merge requires:

- exactly one report for every arm;
- matching experiment, ADR, protocol, authorization, corpus, source, runtime,
  model configuration, and parameter count;
- 500 optimizer steps and 16,000 sampled training examples;
- full 50,000-row train and 10,000-row validation evaluation;
- identical source, D6, and target semantic digests;
- all integrity flags;
- finite metrics;
- positive compile, inference, training, and same-host calibration timing; and
- explicit nonpromotion claims.

For this Stage 2 value screen, a compact arm is a candidate when:

```text
validation total MAE delta <= 1.0
validation total RMSE delta <= 1.5
validation mean component MAE delta <= 0.25
and
(
    same-host inference speedup >= 1.5
    or same-host gradient-step speedup >= 1.3
)
```

Among passing compact arms, the smallest tensor shape is selected, followed by
lower validation total MAE and higher same-host inference speed.

Historical 441 is always diagnostic and can never be selected.

Forward and reverse report orders must produce byte-identical aggregate JSON.

## Claims Boundary

Stage 2 completion can establish:

- exact semantic preservation at the MLX boundary;
- iso-architecture value-learning behavior;
- hardware-normalized shape throughput;
- shape-specific compile and memory cost; and
- one candidate for the next R0 gate.

It cannot establish:

- complete-action ranking quality;
- validation target recall;
- retained regret;
- realistic legal-set action latency;
- paired gameplay noninferiority;
- promotion into the player; or
- progress toward 100 mean.

Those remain mandatory Stage 3 blockers. Every arm report and aggregate sets
`promotion_authorized` and `progress_to_100_claimed` to false.

## Consequences

The tournament is reproducible, resumable, content-addressed, and runnable on
the local four-Mac cluster without external compute. A representation cannot
win by clipping, changing data, changing model capacity, running fewer steps,
or exploiting a faster host.

The explicit 23-row exact control may outperform every bounded dense arm. That
is an admissible and useful result. The experiment measures truth rather than
assuming that a radius grid is faster merely because 61, 91, or 127 is smaller
than the historical 441.

Production execution remains blocked until the parent reviews this ADR and the
preregistration, builds and fans out one immutable bundle, creates the corpus
lock and authorization, reviews the generated queue specification, and
explicitly applies it.

## Outcome

The production tournament completed on 2026-06-17.

All three bounded compact arms were value-noninferior to the 23-row exact
entity control:

| Arm | Validation total MAE delta | Same-host inference ratio | Same-host training ratio |
|---|---:|---:|---:|
| Radius 6 / 150 rows | +0.002905 | 0.140x | 0.143x |
| Radius 5 / 114 rows | +0.002905 | 0.202x | 0.251x |
| Radius 4 / 84 rows | +0.003063 | 0.291x | 0.297x |

None passed the 1.5x inference or 1.3x training leverage gate. The selected
Stage 2 candidate is therefore null. Exact entities remain the R0 control and
preferred substrate. No R1 dense-capacity reinvestment arm is authorized.

Historical 441 was diagnostic only. Exact entities delivered 44.09x its
same-host calibrated inference throughput and 48.03x its calibrated training
throughput while using a small fraction of its active memory.

The next authorized representation work is the R2 matched sparse MLX
architecture tournament. Complete-action ranking and gameplay promotion
remain unmeasured.

Full result:
`docs/v2/reports/r0-spatial-mlx-tournament-v1-result.md`.
