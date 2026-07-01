# ADR 0132: Feature Census Deterministic Host Rebalance

Date: 2026-06-16

Status: accepted before production F1 census launch

Experiment ID: `feature-schema-activation-census-v1`

## Context

ADR 0129 freezes scientific evidence ownership as:

```text
BLAKE3(evidence_id) mod 4
```

Its initial operational table mapped shard index 1 to john2. At launch time,
john2 was already running the sole authorized 200-epoch
`conditional-tile-local-geometry-dropout-v1` MLX origin. The cluster
coordinator permits one research task per host so long jobs cannot silently
oversubscribe memory, GPU, or thermal capacity.

Waiting for john2 would leave three completed, merge-blocked F1 shards for
roughly an hour. Running a second task on john2 would interfere with a
higher-value preregistered origin. Neither outcome improves research
throughput.

Host identity is not part of F1 scientific evidence identity. The shard
index, immutable input manifests, scanner source, and scanner configuration
are the scientific contract.

## Decision

Before any production F1 shard starts:

- shard 0 runs on john1;
- shard 2 runs on john3;
- shard 3 runs on john4; and
- shard 1 runs on john1 after shard 0 completes.

All four tasks use one checksum-verified immutable scanner/source bundle and
the same frozen data roots. Each evidence ID remains owned by exactly one
shard index. No row, cache batch, dataset shard, or legacy shard is duplicated
or reassigned between shard indices.

The queue records the actual host for operational provenance. The scientific
hash continues to exclude host identity, timestamps, absolute paths, and
output paths.

## Consequences

- john2 continues the higher-value MLX origin without interference.
- john1 performs two disjoint F1 shards sequentially.
- john3 and john4 each perform one disjoint F1 shard concurrently.
- final merge still requires the exact set `0/4`, `1/4`, `2/4`, and `3/4`.
- duplicate evidence remains a hard failure.
- this amendment changes no schema, feature semantics, input domain, success
  gate, closed-domain rule, or scientific hash definition.

Any later host rebalance requires the same conditions: it must occur before
the affected shard starts, preserve the shard index and immutable bundle, and
be recorded in the queue and experiment ledger.

