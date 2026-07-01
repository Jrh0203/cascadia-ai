# ADR 0152: S4 Candidate-Context Cache

Status: accepted; production cache complete

Date: 2026-06-17

Experiment: `s4-candidate-context-cache-v1`

Research-plan item: S4

## Context

ADR 0151's corrected source-frozen census authorized 256 observable anchors
and six exact candidate relations. The first S4 neural comparison needs those
relations for every retained train candidate and every complete validation
candidate.

Recomputing relation graphs inside each model arm or training epoch would
waste CPU, create multiple opportunities for semantic drift, and make matched
architecture comparisons harder to audit. Materializing all 6,154,696
overlapping validation edges would also defeat the linear-memory design.

## Decision

Build one immutable, content-addressed context sidecar shared by every S4
model arm.

For each decision group, persist:

- exact action hashes in R3 candidate order;
- the first 256 candidates in stable observable screen-rank order;
- up to eight stable anchor neighbors for each of the six exact ADR 0151
  relations;
- the number of matching anchor siblings for each relation; and
- the selected candidate index for audit and metric binding.

Every index is group-relative `uint16`; `65535` is the missing-index sentinel.
Every complete candidate remains a query. The cache does not shortlist,
remove, reorder, label, or score candidates.

The sidecar uses a deterministic binary container with:

- canonical JSON metadata;
- 64-byte tensor alignment with verified zero padding;
- fixed dtype, shape, byte-count, offset, and BLAKE3 descriptors;
- memory-mappable tensor payloads;
- a content ID over scientific identity and tensor descriptors; and
- exact action-hash binding at load time.

## Source And Data Boundary

Production export must use:

- the corrected ADR 0151 aggregate report
  `2b977892c9b899d2fb9b38cfeb1b2e10c9a4f778650cf68dbadc78b28a33c7fc`;
- the ADR 0150 open-data authorization;
- R3 cache
  `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156`;
- the complete open train and validation datasets; and
- one immutable source bundle built specifically for this cache export.

No sealed test data, gameplay result, hidden bag order, future refill, model
prediction, or R3 classifier output is an input.

## Distributed Execution

The same modulo-three split used by the corrected foundation is retained:

| Host | Rows |
|---|---|
| john2 | `row % 3 == 0` |
| john3 | `row % 3 == 1` |
| john4 | `row % 3 == 2` |

Each host writes one deterministic self-describing container. John1 collects
the containers by checksum and merges them in canonical row order. A second
merge with reversed shard arguments must produce the same cache ID and
container bytes.

## Acceptance

Classify `context_cache_ready` only if:

1. the source bundle is whole-tree identical on every worker;
2. all 560 train and 240 validation rows occur exactly once;
3. validation contains exactly 860,203 complete candidates;
4. every action-hash vector matches the R3 cache order;
5. every stored context index passes its tensor and bounds contract;
6. all tensor and container checksums verify;
7. forward and reverse merges are byte-identical; and
8. the final memory-mapped loader binds every split and rejects action drift.

Otherwise classify `context_cache_invalid` and do not launch S4 training.

## Consequences

1. Every S4 neural arm receives byte-identical candidate context.
2. Relation construction cost is paid once and removed from training epochs.
3. Exact semantics remain inspectable independently of MLX model code.
4. Bounded relation neighbors keep storage and inference linear in candidate
   count.
5. The cache can be reused by 256-anchor and 128-anchor serving ablations
   without rebuilding the open corpus.

## Result

The production source bundle
`ebd41fbed3ca8009d53e9bad06b194b6043fc6021e791b5c4dd95ca91254a621`
was whole-tree verified on john2, john3, and john4. The three disjoint shards
covered 280,012 retained train candidates and all 860,203 validation
candidates.

Forward and reverse merges produced the same cache ID and byte-identical
container and manifest:

```text
fd3dcc8018cfe4b735a9a6514555e90e938fd142e746dc6d791f482e96463def
```

The memory-mapped context container is 166,903,850 bytes. A fresh strict
loader traversed every group. An independent full R3 binding audit matched
every group ID, candidate offset, selected index, and action hash, producing
audit ID
`fee5eca1add3db2402b8c160dd71246e12a7ab3cba2158a89bf8cbc8492525da`.

The final merged cache was then copied to its canonical production path on
john2, john3, and john4. Whole-tree verification on every worker matched both
required files:

```text
context.s4ctx  bfa6515cb7797b5b5d50f16c5d73e5a23d050df8db3959f299143f0ac1c6660a
cache.json     4ee309ac61c4bf64d08cde2d66973724199a9f41f2c0345415227e7a87c12639
```

The fanout receipt is
`artifacts/experiments/s4-candidate-context-cache-v1/reports/production-cache-fanout.json`.

The terminal classification is:

```text
context_cache_ready
```

See `docs/v2/reports/s4-candidate-context-cache-v1-result.md`.
