# ADR 0138: R0 Production Corpus And Execution Graph

Status: accepted

Date: 2026-06-16

Experiment: `r0-spatial-footprint-screen-v1`

## Context

R0 requires more than a benchmark binary. Its Stage 1 corpus must contain at
least 50,000 open-train and 10,000 open-validation `PositionRecord` rows, use
disjoint complete-game intervals, remain byte-identical on all four Macs, and
feed four nonoverlapping ordinal partitions. Its timing protocol separately
requires at least three independent release-process invocations per partition.

Ad hoc SSH commands would make retries, source identity, dependencies, and
replicate completeness difficult to audit. A single large shared dataset
writer would also serialize collection and leave most of the cluster idle.

## Decision

`tools/r0_spatial_campaign.py` generates and atomically installs the complete
queue graph after an immutable Rust experiment bundle exists. The bundle must
contain:

- `cascadia-v2`;
- `spatial_representation_benchmark`;
- `bundle.json`; and
- every selected Rust source file and lockfile required to reproduce them.

It must also contain the complete source roots hashed by
`cascadia-provenance`, including `CASCADIA_V2_GOAL.txt`, the workspace and
Python lock/configuration files, MLX package sources, web sources, and every
V2/legacy Rust source root named by the provenance crate. A bundle that cannot
reconstruct the collector's source digest is rejected before queue mutation.

The campaign tool validates the bundle before producing any task. It rejects a
bundle outside the repository, missing binaries, fewer than three timing
replicas, nonpositive benchmark iterations, duplicate generated task IDs, or a
queue that already contains any generated ID.

## Production Corpus

Every game contributes 80 positions. Collection is split by complete,
contiguous, nonoverlapping game-index intervals:

| Host | Train first | Train games | Train rows | Validation first | Validation games | Validation rows |
|---|---:|---:|---:|---:|---:|---:|
| john1 | 200000 | 157 | 12560 | 210000 | 32 | 2560 |
| john2 | 200157 | 156 | 12480 | 210032 | 31 | 2480 |
| john3 | 200313 | 156 | 12480 | 210063 | 31 | 2480 |
| john4 | 200469 | 156 | 12480 | 210094 | 31 | 2480 |
| Total | | 625 | 50000 | | 125 | 10000 |

Every part uses:

- four-player AAAAA;
- habitat bonuses disabled by the canonical V2 game contract;
- `pattern-aware`;
- the declared train or validation split;
- eight games per physical shard where possible;
- `--resume` for an interrupted writer; and
- no test or final data.

The queue launches every collector through `/usr/bin/env -C` with the
immutable bundle's `source/` directory as its working repository and the
bundle's own executable by absolute path. Dataset outputs remain under the
host's normal artifact root. This makes the source digest independent of
whether a worker has a full checkout, a partial checkout, or no checkout
outside the immutable bundle.

Each host writes its own directory. The coordinator then retrieves or validates
that complete tree and fans it to john2, john3, and john4 with whole-tree
SHA-256 equality. Benchmarking does not begin until all eight trees are
identical everywhere.

## Mechanical Screen

Each Mac owns one deterministic modulo partition:

```text
global_ordinal % 4 == shard_index
```

Every host runs all five R0 arms in the same process so comparisons remain
paired on the same records, hardware, load, and thermal context. Each
partition is executed in three separate processes with explicit replicate
indices `0`, `1`, and `2`. The queue never treats loop iterations inside one
process as independent replication.

The resulting 12 reports are:

- three on john1 for shard 0;
- three on john2 for shard 1;
- three on john3 for shard 2; and
- three on john4 for shard 3.

Remote reports are collected with source and destination SHA-256 proof. The R0
classifier consumes all 12 reports in forward and reverse order. The two
terminal aggregates must be byte-identical.

## Queue Semantics

The graph contains 33 tasks:

- one immutable bundle fanout;
- eight independent dataset collections;
- eight whole-tree dataset fanouts;
- twelve independent process benchmarks;
- one remote report collection;
- two independently ordered classifications; and
- one byte-level merge-order proof.

Lower numeric priority follows the dependency chain. john2 remains occupied by
the already authorized local-geometry dropout origin until that task releases
the host; john1, john3, and john4 can begin their nonduplicative collection
parts immediately. The scheduler still permits at most one queued task per host
at a time, while each collection or benchmark process may use the host's
declared CPU budget internally.

## Claims

This graph can establish:

- exact corpus size and split integrity;
- byte-identical source data on all workers;
- exact shard and replicate coverage;
- lossless round trips and semantic identity;
- extraction, serialization, deserialization, and storage ratios; and
- deterministic aggregation.

It cannot establish learned representation sufficiency or gameplay strength.
Those remain R0 Stage 2 and Stage 3 decisions.
