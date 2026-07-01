# ADR 0135: R0 Lossless Spatial Representation Contract

Status: accepted

Date: 2026-06-16

Contract ID: `r0-lossless-spatial-representation-v1`

## Context

F2 established that the historical 441-cell lattice is materially larger than
the measured AAAAA board support:

- a complete centered radius-4 disk contains 61 cells;
- a complete centered radius-5 disk contains 91 cells;
- a complete centered radius-6 disk contains 127 cells;
- no complete centered hex disk contains 121 cells;
- recentered radius 6 retained every measured generated event;
- recentered radius 5 retained every occupied entity and nearly every measured
  frontier or selected destination; and
- legal straight and bent 23-tile boards still exceed radius 6.

The empirical radius-6 result does not authorize clipping. The rules domain
and the adversarial controls require an exact overflow path in every bounded
arm.

F3 separately established the authoritative D6 coordinate and tile-orientation
contract in `cascadia-game`. R0 may bind that contract but may not redefine
rotation, reflection, edge order, or dual-terrain orientation.

The representation tournament therefore needs one Rust-owned substrate that
changes only spatial indexing. It must preserve the existing
`compact-entity-v2` observables exactly, expose stable accounting, and permit
the MLX arms to compare 127, 91, 61, 441, and exact entities without hidden
semantic differences.

## Decision

`cascadia-data::spatial_representation` owns the R0 V1 spatial contract.

The permanent arm order and exact CLI IDs are:

| R0 arm | Rust variant | CLI ID | Local support |
|---|---|---|---:|
| R0-A | `ExactEntityControl` | `exact-entity-control` | sparse exact coordinates |
| R0-B | `HexRadius6` | `hex-radius-6-127` | complete radius-6 disk, 127 rows |
| R0-C | `HexRadius5` | `hex-radius-5-91` | complete radius-5 disk, 91 rows |
| R0-D | `HexRadius4` | `hex-radius-4-61` | complete radius-4 disk, 61 rows |
| R0-H | `HistoricalSquare21` | `historical-square-21x21-441` | fixed-origin 21x21 axial square |

Unknown IDs and duplicate repeated IDs fail deterministically. Omitting
`--arm` selects all arms in the table order. Repeated `--arm` arguments select
a subset, but the report and execution order remain canonical rather than
depending on argument order.

## Semantic Invariant

Every occupied tile entity carries the same six raw semantic channels in the
same order:

1. primary terrain;
2. secondary terrain or `NONE`;
3. canonical tile rotation;
4. allowed-wildlife bit mask;
5. placed wildlife or `NONE`; and
6. keystone flag.

The two coordinate bytes are the only fields replaced by local indexing.
Turn, perspective, player count, board counts, Nature Tokens, scoring cards,
habitat-bonus flag, per-player wildlife counts, per-player largest habitats,
market entities, and targets remain byte-equivalent across arms.

Extraction rejects:

- invalid terrain, rotation, wildlife, mask, or keystone values;
- noncanonical rotation on a single-terrain tile;
- wildlife absent from the tile compatibility mask;
- keystone entities with a secondary terrain;
- coordinates outside the V2 rules backing grid;
- duplicate occupied coordinates;
- nonempty padding rows; and
- board counts above the 23-tile rules limit.

The invariant is:

```text
exact rows + active local rows + exact overflow rows
    == source occupied rows
```

No arm has a drop, clip, pad-as-overflow, or aggregate-only path.

## Recentered Hex Frames

R0-B, R0-C, and R0-D use the exact F2 minimax integer-center algorithm.

For occupied axial coordinates, the algorithm finds the minimum possible
maximum hex radius. When multiple integer centers attain that radius, it uses
the stable F2 `(q, r)` search order. Empty boards use `(0, 0)`.

The center is serialized with each bounded hex board. Original coordinates are
reconstructed exactly as:

```text
absolute = local_relative_coordinate + center
```

The historical 21x21 arm remains fixed at the rules origin. Recentring it
would no longer isolate the historical shape.

The integer tie-break is deterministic but is not falsely declared to be an
independent D6-equivariant selector for every symmetric tie. D6 augmentation
must transform the already selected center together with the entities. It may
not transform a state and independently rerun the tie-break.

## Stable Local Indexing

Complete disks enumerate axial rows in ascending `q`, then ascending `r`
within the exact radius constraint:

```text
max(abs(q), abs(r), abs(-q-r)) <= radius
```

Disk indices fit in one byte. The historical square enumerates ascending `q`,
then ascending `r`, over `[-10, 10] x [-10, 10]` and uses a two-byte index.

Each index has an exact inverse. The implementation tests every index in all
four bounded supports.

## Exact Overflow

An entity outside local support is serialized as:

```text
absolute q, absolute r, six unchanged semantic channels
```

Overflow entities are sorted by absolute coordinate and remain individually
addressable. The path is not a pooled token and does not discard multiplicity,
orientation, wildlife compatibility, or placed wildlife.

The radius-6, radius-5, radius-4, and historical-square arms all use this same
overflow entity schema. Later MLX models may additionally derive an overflow
summary token, but the exact entities remain authoritative.

## D6 Contract

`D6Transform` from `cascadia-game` transforms:

- absolute coordinates;
- the carried recentering coordinate;
- local relative coordinates;
- dual-terrain tile rotation; and
- single-terrain rotation, canonicalized to zero.

Complete hex disks are D6-closed. Every local disk index therefore has an
exact local-index permutation for each of the 12 transforms.

The historical axial square is not D6-closed. A square-local entity can become
overflow, and an overflow entity can become square-local, after transformation.
The implementation reconstructs the exact coordinate, applies the rules-owned
transform, and reindexes without clipping.

A transform that would leave the finite V2 rules backing grid fails explicitly,
matching the rules-owned board transform boundary. Within supported rules
coordinates:

```text
decode(transform(encode(state))) == transform(state)
transform_inverse(transform(encoding)) == encoding semantics
```

For states with a unique minimax center, fresh extraction after transformation
also equals the transformed compact encoding. Tied states use the carried-frame
rule above.

## Packed V1 Format

The self-contained packed representation uses:

- magic `CSR0SP1\0`;
- schema version `1`;
- one stable arm code;
- one reserved zero byte;
- the unchanged `PositionRecord` metadata prefix;
- four arm-specific board payloads; and
- the unchanged market and target suffix.

Board payloads are:

| Arm | Board payload |
|---|---|
| exact | one 8-byte exact entity per `board_count` |
| radius 6/5/4 | 2-byte center, 1-byte local count, 7-byte local rows, 8-byte overflow rows |
| historical 441 | 1-byte local count, 8-byte local rows, 8-byte overflow rows |

The packed decoder validates magic, schema, arm, reserved bytes, counts,
indices, ordering, semantic channels, overflow classification, exact
coordinates, and trailing bytes. Decode followed by `PositionRecord`
reconstruction must equal the source record.

`SpatialRepresentationAccounting` reports:

- total packed bytes;
- packed spatial bytes;
- local capacity rows;
- active local rows;
- exact entity rows;
- exact overflow rows;
- total semantic entity rows; and
- dense raw scalar slots including an occupancy channel.

The packed-byte result describes the Rust V1 serialization contract. MLX tensor
memory depends on dtype and feature expansion and must be reported separately.

## Benchmark Interface

The production benchmark binary is:

```text
cargo run --release -p cascadia-data \
  --bin spatial_representation_benchmark -- ...
```

Required input:

```text
--dataset-root PATH
--replicate-index R
```

Repeatable selection:

```text
--arm exact-entity-control
--arm hex-radius-6-127
--arm hex-radius-5-91
--arm hex-radius-4-61
--arm historical-square-21x21-441
```

Deterministic record partition:

```text
--shard-index I --shard-count N
```

`--replicate-index` is required and accepts exactly `0`, `1`, or `2`. These
values identify the three independent classifier-eligible process invocations
for each shard. Iterations inside one process are not independent replicates.

Shard defaults are `0/1`. The global record ordinal is defined by:

1. `--dataset-root` argument order;
2. manifest shard order; and
3. in-shard record order.

A record is eligible exactly when:

```text
global_ordinal % shard_count == shard_index
```

Every manifest and shard is validated before partition interpretation. The
`--records` limit is applied after modulo selection. `--records 0` means every
eligible row.

The report includes:

- replicate index;
- selected arm IDs;
- shard index and count;
- the exact ordinal rule;
- short hostname, `std::env::consts` OS and architecture, logical parallelism,
  CPU brand, memory bytes, and a stable hardware description;
- total manifested, eligible, and loaded record counts;
- per-dataset global ordinal ranges and eligible/loaded counts;
- source and per-arm semantic BLAKE3;
- exact round-trip status;
- extraction, serialization, and deserialization timing;
- packed size and ratio to the 864-byte source record;
- local capacity and occupancy;
- exact and overflow row counts; and
- dense raw scalar-slot accounting.

Missing, duplicate, or out-of-range replicate indices, invalid `I/N`, unknown
arms, duplicate arms, incompatible datasets, checksum drift, empty selections,
or semantic drift fail nonzero.

Execution provenance is operational attribution only. Hostname and hardware
fields are excluded from semantic equality, source digests, and packed
round-trip identity. On macOS, CPU and memory use best-effort
`sysctl -n machdep.cpu.brand_string` and `sysctl -n hw.memsize`; failures are
reported as `unknown` without changing benchmark semantics.

## Required Verification

The permanent Rust suite covers:

- exact 61, 91, 127, and 441 capacities;
- every local index and inverse;
- all 12 D6 permutations for every disk row;
- deterministic center ties;
- carried-center transformation;
- exact in-memory and packed round trips for every arm;
- legal straight and bent 23-tile radius-6 overflow controls;
- historical-square local-to-overflow transformation;
- transformed tile orientation;
- inverse D6 round trips;
- truncated and trailing packed bytes;
- canonical arm selection and duplicate/unknown rejection;
- required replicate indices and duplicate/out-of-range rejection;
- invalid shard parameters;
- modulo-shard disjointness and complete union; and
- post-partition record limiting.

## Consequences

R0 model arms can share one exact Rust source contract. A compact model cannot
claim a speedup by silently changing public semantics.

The MLX implementation remains responsible for:

- decoding these arms into matched semantic feature channels;
- adding exact frontier, action, component, motif, and relation tokens where
  the later R0 model requires them;
- preserving the exact overflow stream rather than replacing it with only a
  summary;
- binding the generated D6 row permutations or group averaging;
- reporting compile time, throughput, latency, and peak memory by dtype; and
- proving offline and gameplay noninferiority under the R0 preregistration.

Any change to arm IDs, index order, center selection, overflow semantics,
packed fields, channel order, or D6 behavior requires a schema bump and an ADR
amendment.
