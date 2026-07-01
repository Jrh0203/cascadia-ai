# R0 Spatial-Footprint Screen Preregistration

Date: 2026-06-16

Experiment ID: `r0-spatial-footprint-screen-v1`

Contract: ADR 0135, `r0-lossless-spatial-representation-v1`

Goal: select the smallest lossless spatial substrate that materially improves
research throughput without degrading decision quality or playing strength on
four-player AAAAA with habitat bonuses disabled.

## Hypotheses

### R0-A: Exact entity control

Sparse exact-coordinate tile entities are the scientific control. They preserve
the complete current V2 coordinate domain without materializing a 2,401-cell
dense tensor.

### R0-B: Radius 6 / 127

A recentered complete radius-6 disk plus exact overflow will retain the
measured corpus almost entirely in its local path and materially reduce dense
spatial work without changing semantics.

### R0-C: Radius 5 / 91

A recentered complete radius-5 disk plus exact overflow will move a small
fraction of frontier or action-adjacent states through overflow while reducing
local spatial cost further.

### R0-D: Radius 4 / 61

A recentered complete radius-4 disk plus exact overflow is an aggressive
locality hypothesis. It is promoted only if the exact overflow path preserves
decision quality; occupancy coverage alone is insufficient.

### R0-H: Historical 21x21 / 441

The fixed-origin historical axial square is a diagnostic, not the control. It
tests whether any observed advantage came from the old shape rather than
semantic quality. It must retain exact overflow.

## Frozen Arm IDs

| Arm | CLI ID |
|---|---|
| R0-A | `exact-entity-control` |
| R0-B | `hex-radius-6-127` |
| R0-C | `hex-radius-5-91` |
| R0-D | `hex-radius-4-61` |
| R0-H | `historical-square-21x21-441` |

No aliases are accepted in V1.

## Controlled Variables

All learned arms must use:

- identical public states and targets;
- identical train, validation, and later paired-game seeds;
- identical semantic tile channels;
- identical nonspatial global features;
- identical exact overflow entities;
- identical action candidates and legal masks;
- identical D6 transform IDs and schedule;
- identical optimizer, dtype, batch schedule, hidden width, parameter budget,
  initialization policy, and training steps;
- identical checkpoint selection; and
- identical evaluation code.

Only spatial indexing, local support shape, and the resulting exact
local-versus-overflow partition may differ.

No arm may receive hidden stack order, hidden wildlife order, realized future
refills, future actions, or terminal outcomes as model input.

## Evidence Stages

### Stage 0: Rust mechanical qualification

Before MLX work, every arm must pass:

- exact `PositionRecord` in-memory round trip;
- exact packed encode/decode round trip;
- source-equal semantic BLAKE3;
- exact entity-count conservation;
- deterministic indexing and recentering;
- legal straight and bent overflow controls;
- D6 local-index permutation for complete disks;
- exact historical-square reclassification through overflow;
- inverse D6 reconstruction;
- deterministic arm selection; and
- disjoint, complete modulo record sharding.

Failure blocks that arm from training.

### Stage 1: Manifested extraction screen

Create immutable, checksummed, open `compact-entity-v2` position datasets for
R0. The corpus must include:

- four-player AAAAA;
- habitat bonuses disabled;
- all four relative player boards;
- opening, early, middle, and late states;
- current pattern-aware play;
- at least 50,000 train positions;
- at least 10,000 validation positions; and
- no test or final split.

The collection manifest, every shard checksum, source identity, executable
identity, strategy ID, split, game-index interval, and record total must be
frozen before timing.

The extraction binary accepts only the base `DatasetManifest` /
`PositionShardReader` contract. Wrapped score-to-go, ranking, imitation, and
graded-oracle records are not silently reinterpreted as base shards.

### Stage 2: Matched MLX representation screen

Implement one matched small model per mechanically qualified arm. The first
screen is deliberately capacity-limited: it asks whether the representation
is sufficient, not how much architecture can mask its defects.

Required outputs:

- train and validation loss;
- tie-aware top-one target recall;
- confidence-set mass;
- mean retained regret;
- high-regret tail;
- value calibration where applicable;
- compile time;
- examples per second;
- complete-action rows per second;
- median, P95, and P99 inference latency;
- peak resident and MLX memory;
- packed bytes;
- local capacity and active rows;
- overflow rows and overflow-state fraction; and
- D6 consistency error.

### Stage 3: Limited gameplay qualification

Only arms passing the offline gates receive a paired 20-game smoke. Games use
the frozen symmetric four-player AAAAA configuration with no habitat bonuses.
The exact control and treatment use common random numbers and seat rotation.

No 50-, 100-, or final-game confirmation is authorized by this
preregistration.

## Cluster Partition Contract

The extraction stream has one stable global ordinal:

1. dataset roots in CLI order;
2. shards in manifest order;
3. rows in shard order.

For shard `I/N`, a record belongs to the host exactly when:

```text
global_ordinal % N == I
```

Partitioning occurs after every manifest and shard has validated and before the
local `--records` limit. The primary full-corpus run uses `--records 0`.

Every shard is measured by three independent process invocations identified by
the required `--replicate-index 0`, `--replicate-index 1`, and
`--replicate-index 2`. The classifier accepts exactly one report for each of
those indices per shard and arm. Iterations within one process are repeated
measurements, not independent replicates.

The recommended four-host extraction wave is:

| Host | Partition | Arms |
|---|---|---|
| john1 | `--shard-index 0 --shard-count 4` | all five |
| john2 | `--shard-index 1 --shard-count 4` | all five |
| john3 | `--shard-index 2 --shard-count 4` | all five |
| john4 | `--shard-index 3 --shard-count 4` | all five |

This is nonduplicative evidence: each host sees distinct records, while each
host times the exact control and every treatment on the same hardware and
records. Per-host treatment/control ratios therefore normalize chip and load
differences.

If a host cannot run all arms in one task, repeated `--arm` selects a canonical
subset. Every arm must still cover every ordinal partition before the
scientific report closes.

Example full partition:

```bash
cargo run --release -p cascadia-data \
  --bin spatial_representation_benchmark -- \
  --dataset-root artifacts/datasets/r0-spatial-position-corpus-v1-train \
  --dataset-root artifacts/datasets/r0-spatial-position-corpus-v1-validation \
  --replicate-index 0 \
  --shard-index 2 \
  --shard-count 4 \
  --records 0 \
  --iterations 50 \
  --output artifacts/experiments/r0-spatial-footprint-screen-v1/runs/john3-shard2-replicate0.json
```

Example independent R0-C task with a same-host control:

```bash
cargo run --release -p cascadia-data \
  --bin spatial_representation_benchmark -- \
  --dataset-root artifacts/datasets/r0-spatial-position-corpus-v1-train \
  --dataset-root artifacts/datasets/r0-spatial-position-corpus-v1-validation \
  --arm exact-entity-control \
  --arm hex-radius-5-91 \
  --replicate-index 0 \
  --shard-index 1 \
  --shard-count 4 \
  --records 0 \
  --iterations 50
```

## Timing Protocol

Use release binaries. Record:

- binary BLAKE3;
- V2 source BLAKE3;
- dataset manifest BLAKE3 values;
- host identifier and chip;
- macOS version;
- physical memory;
- power mode;
- process concurrency;
- thermal state before and after;
- selected arms;
- replicate index;
- shard metadata; and
- iteration count.

The benchmark performs one untimed warmup extraction per loaded record, then
times extraction, packed serialization, and packed deserialization separately.
Every report also captures short hostname, standard-library OS and
architecture, logical parallelism, CPU brand, memory bytes, and a stable
hardware description. These fields attribute operational measurements and are
never included in semantic equality or arm digests. Failed best-effort macOS
`sysctl` probes are recorded explicitly as unknown.

Run exactly the three classifier-eligible independent process invocations
`--replicate-index 0`, `1`, and `2` per host partition. Report the median
invocation for absolute latency and all three invocations for variance. Do not
treat iterations within one process as independent samples. Additional
exploratory runs must use a separate experiment identity and cannot be merged
into this preregistered classifier input.

For each host and treatment, calculate:

```text
extraction_speedup = exact_ns_per_record / treatment_ns_per_record
packing_speedup = exact_pack_ns_per_record / treatment_pack_ns_per_record
decode_speedup = exact_decode_ns_per_record / treatment_decode_ns_per_record
```

Aggregate counts and elapsed nanoseconds across disjoint shards. Do not average
host means without weighting by timed extraction count. Report both the
weighted global result and the distribution of same-host ratios.

## Integrity And Merge Gates

The combined extraction report is valid only when:

- every expected dataset manifest and shard checksum passes;
- the union of global ordinals is complete;
- pairwise shard intersections are empty;
- every selected arm reports the source semantic BLAKE3;
- every arm covers the same expected global ordinal set;
- every arm passes both round trips;
- loaded and eligible counts reconcile;
- selected arm IDs are canonical;
- source, executable, and packed schema identities agree; and
- no report contains NaN, infinity, zero timed operations, or an empty shard.

Wall-clock timestamps and output paths are provenance, not scientific payload.

## Mechanical Success

The Rust foundation passes when:

- all focused library and benchmark tests pass;
- `cargo fmt --all -- --check` passes for the owned Rust files;
- `cargo clippy -p cascadia-data --all-targets -- -D warnings` passes;
- a current `compact-entity-v2` manifested smoke dataset validates;
- every arm completes a release benchmark smoke;
- selected-arm reports contain exactly the requested canonical set;
- four-way shard metadata reconciles; and
- the help output contains no patch artifacts or undocumented arm IDs.

## Representation Promotion Gates

A compact arm remains eligible for R1 only if it:

1. preserves 100% of exact source and legal-action semantics through local
   plus overflow representation;
2. passes every adversarial and D6 probe;
3. delivers at least 1.5x state-build or model throughput, or at least 30%
   end-to-end training/search throughput, against R0-A;
4. keeps validation target recall within 1 percentage point of R0-A;
5. increases mean retained regret by no more than 0.02;
6. does not materially worsen high-regret tail behavior;
7. shows no paired 20-game degradation large enough to trigger the frozen
   futility boundary; and
8. does not gain speed by omitting exact overflow processing.

R0-H cannot win by compatibility alone. If a compact arm is noninferior to
R0-A and faster than R0-H, the 441-cell shape is retired from new model work.

## Decision Rules

- If R0-B passes quality and throughput gates, it becomes the default dense
  compact control for R1.
- If R0-C also passes and is materially faster, R0-C supersedes R0-B.
- R0-D is promoted only on full quality evidence; high overflow incidence
  alone does not reject it because overflow is exact.
- If no dense compact arm passes, retain R0-A and proceed to sparse
  occupied-plus-frontier R2.
- A result that changes semantic channels, model capacity, target corpus, D6
  schedule, or overflow information is confounded and cannot choose an arm.

## MLX Handoff And Current Blockers

ADR 0135 intentionally stops at the Rust source and benchmark boundary. The
later MLX arm still needs:

1. a decoder for the V1 packed schema or a directly equivalent Rust-to-MLX
   tensor adapter;
2. matched tensor layouts for exact, 127, 91, 61, and 441 supports;
3. exact overflow batching with masks and no pooled-only substitution;
4. rules-generated disk D6 row permutations;
5. action, legal-frontier, component, motif, and relation features that use
   the same spatial indexer;
6. one state trunk shared across legal actions where the architecture permits;
7. fixed parameter, optimizer, dtype, and training manifests;
8. end-to-end MLX compile, throughput, latency, and memory instrumentation;
9. open-validation decision metrics; and
10. paired gameplay qualification.

The current `PositionRecord` contains occupied tile entities and global board
summaries, but not an explicit legal-frontier stream, selected action token,
habitat-component identity, wildlife-motif identity, or complete-action
relation graph. Those are model-side R0/R2 inputs still to be implemented.
They may use this spatial contract but may not alter it retroactively.

## Stop Conditions

Stop and classify the run invalid if:

- any entity is clipped or summarized without its exact row;
- a host receives overlapping ordinals under the same evidence wave;
- an arm is missing a required partition;
- a checksum or semantic digest differs;
- a compact arm uses a different semantic channel set;
- a D6 transform independently reselects a tied center;
- the sealed test or final split is opened;
- hidden information enters the model input; or
- gameplay begins before offline qualification.

This preregistration selects a representation substrate. It does not claim a
100-point player result by itself.
