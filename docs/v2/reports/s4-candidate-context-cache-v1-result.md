# S4 Candidate-Context Cache V1 Result

Date: 2026-06-17

ADR: 0152

Experiment: `s4-candidate-context-cache-v1`

Status: complete

Classification: `context_cache_ready`

## Verdict

The exact S4 candidate-context cache is ready for matched neural training.
Every future S4 arm can consume the same memory-mapped 256-anchor and bounded
six-relation sidecar without rebuilding relation graphs during training.

The cache covers all retained open train candidates and every complete open
validation candidate. It does not prune candidates, read sealed data, expose
hidden state, or contain model predictions.

## Immutable Evidence

| Identity | Value |
|---|---|
| Production source bundle | `ebd41fbed3ca8009d53e9bad06b194b6043fc6021e791b5c4dd95ca91254a621` |
| Corrected S4 foundation report | `2b977892c9b899d2fb9b38cfeb1b2e10c9a4f778650cf68dbadc78b28a33c7fc` |
| R3 cache | `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156` |
| Merged context cache | `fd3dcc8018cfe4b735a9a6514555e90e938fd142e746dc6d791f482e96463def` |
| Merge-order proof | `79f17f580c5bf8c2e897abe5def708832ba3c2fedb4eb21e0f83cd8d019eec31` |
| Full R3 binding audit | `fee5eca1add3db2402b8c160dd71246e12a7ab3cba2158a89bf8cbc8492525da` |

The context container is 166,903,850 bytes with SHA-256
`bfa6515cb7797b5b5d50f16c5d73e5a23d050df8db3959f299143f0ac1c6660a`.
The manifest is 2,488 bytes with SHA-256
`4ee309ac61c4bf64d08cde2d66973724199a9f41f2c0345415227e7a87c12639`.

## Coverage

| Split | Groups | Candidates |
|---|---:|---:|
| Train | 560 | 280,012 |
| Validation | 240 | 860,203 |
| Total | 800 | 1,140,215 |

The train split is the frozen R3 at-most-512 cohort. Validation is the full
complete-action split.

## Distributed Export

| Host | Shard | Train candidates | Validation candidates | Bytes |
|---|---:|---:|---:|---:|
| john2 | 0 | 92,827 | 290,020 | 56,043,976 |
| john3 | 1 | 93,664 | 283,305 | 55,185,676 |
| john4 | 2 | 93,521 | 286,878 | 55,686,024 |

The production bundle and corrected foundation inputs were whole-tree
verified on all three workers before export. Coordinator collection matched
remote and local SHA-256 for every shard.

## Determinism And Integrity

The cache passed:

- canonical JSON header validation;
- fixed tensor dtype, shape, offset, byte-count, and BLAKE3 validation;
- zero-only 64-byte alignment padding;
- content-address verification;
- exact row coverage with no overlap;
- group-relative index and missing-sentinel validation;
- all 800 selected-index bounds checks;
- strict traversal of all context groups;
- byte-identical forward and reverse merges; and
- byte-identical real-data smoke reruns before production.

The focused suite contains 11 passing tests. The preproduction real-data smoke
caught one verifier-only bug for actions without wildlife placement. The
stored context builder was correct; the verifier incorrectly subtracted an
invalid candidate from a relation class it had not joined. A regression test
was added before the production bundle was frozen.

## Independent R3 Binding

A fresh strict audit re-opened the complete R3 cache with checksum and semantic
verification, then compared the merged context cache against source tensors.

Both splits matched exactly on:

- group IDs;
- candidate offsets;
- every 32-byte action hash; and
- selected source actions converted to group-relative indices.

The audit covered all 1,140,215 cached candidates and produced
`all_bindings_match = true`.

## Production Fanout

The merged production cache was copied to the canonical experiment cache path
on john2, john3, and john4 after classification. Each worker independently
matched the coordinator's complete two-file tree:

| File | Bytes | SHA-256 |
|---|---:|---|
| `context.s4ctx` | 166,903,850 | `bfa6515cb7797b5b5d50f16c5d73e5a23d050df8db3959f299143f0ac1c6660a` |
| `cache.json` | 2,488 | `4ee309ac61c4bf64d08cde2d66973724199a9f41f2c0345415227e7a87c12639` |

The receipt records `all_destinations_match = true` and
`whole_tree_verified = true`:

`artifacts/experiments/s4-candidate-context-cache-v1/reports/production-cache-fanout.json`

## Consequences

1. Use cache
   `fd3dcc8018cfe4b735a9a6514555e90e938fd142e746dc6d791f482e96463def`
   for every arm in the first S4 neural tournament.
2. ADR 0150 subsequently classified every compact R3 treatment as degraded.
   Radius one may now be used only as the explicitly failed substrate in the
   ADR 0153 candidate-context rescue comparison.
3. Materialize relation neighbors as anchor slots at batch time; the immutable
   cache retains auditable group-relative candidate indices.
4. Keep every candidate as a query and use context only as additional
   evidence.
5. Do not regenerate relation graphs inside model training.
