# S4 Candidate-Context Cache V1 Preregistration

Date: 2026-06-17

Status: frozen before production export

Experiment ID: `s4-candidate-context-cache-v1`

Decision: ADR 0152

## Purpose

Create the single immutable exact-relation sidecar consumed by every arm of
the first S4 neural comparison. This is infrastructure qualification, not a
model-strength experiment.

## Frozen Tensor Contract

For each split:

| Tensor | Dtype | Shape |
|---|---|---|
| `rows` | `uint32` | `[groups]` |
| `group_ids` | `uint64` | `[groups]` |
| `candidate_offsets` | `uint64` | `[groups + 1]` |
| `selected_indices` | `uint16` | `[groups]` |
| `action_hashes` | `uint8` | `[candidates, 32]` |
| `anchor_indices` | `uint16` | `[groups, 256]` |
| `relation_neighbor_indices` | `uint16` | `[candidates, 6, 8]` |
| `relation_neighbor_counts` | `uint8` | `[candidates, 6]` |
| `relation_anchor_sibling_counts` | `uint16` | `[candidates, 6]` |

The relation axis is frozen in this order:

1. same draft;
2. same frontier;
3. same tile pose;
4. same wildlife destination;
5. same sibling plan; and
6. equivalent afterstate.

Neighbors are the first eight matching anchors in stable screen-rank order,
excluding self. Counts above eight remain available through
`relation_anchor_sibling_counts`.

## Deterministic Container

Each shard and the merged cache use magic `CSD2S4C\0`, canonical JSON headers,
64-byte tensor alignment, zero-only padding, and raw C-order tensor payloads.
The container ID is BLAKE3 over:

- schema and cache identity;
- scientific provenance;
- every tensor name, dtype, shape, offset, byte count, and BLAKE3 digest.

Two executions with the same inputs must produce byte-identical containers.

## Frozen Inputs

- corrected ADR 0151 aggregate report ID
  `2b977892c9b899d2fb9b38cfeb1b2e10c9a4f778650cf68dbadc78b28a33c7fc`;
- ADR 0150 open-data verification
  `a056aceadb7f53c01dc87c8a39d95a7866bac6df93b050c45cc860de2b8b87ea`;
- R3 cache ID
  `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156`;
- complete open train and validation datasets; and
- a new immutable source bundle containing the exporter, loader, exact
  relation implementation, tests, ADR, and this preregistration.

## Cluster Protocol

Run three disjoint shards:

| Host | Remainder |
|---|---:|
| john2 | 0 |
| john3 | 1 |
| john4 | 2 |

Use modulus three for both splits. Collect each container by coordinator-side
and remote checksum. Merge once as `0,1,2` and once as `2,1,0`.

## Frozen Checks

- complete row coverage with no overlap;
- exactly 860,203 validation candidates;
- source bundle, open-data proof, foundation report, and R3 cache identities
  agree across shards;
- every candidate index and sentinel is in bounds;
- selected indices are valid;
- action hashes bind byte-for-byte to source order;
- tensor payload hashes and whole-container hashes verify;
- no nonzero alignment padding;
- forward and reverse cache IDs match;
- forward and reverse container bytes match; and
- fresh strict loader inspection succeeds.

## Classifications

- `context_cache_ready`: every frozen check passes.
- `context_cache_invalid`: any identity, coverage, tensor, checksum, padding,
  binding, or order-invariance check fails.

No S4 neural run may start from an invalid cache.
