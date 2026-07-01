# Exact MLX Row Locality Order Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

The exact shared-memory MLX path evaluated 5,062,306 sparse rows across 7,709
service requests. Lexicographically ordering the already-deduplicated rows
before encoding them into the existing CSR mapping placed rows with long
common feature prefixes next to one another, with the goal of improving
ordinary first-layer weight-cache reuse inside the unchanged MLX kernels.

The treatment changed no feature within a row, wire metadata, model
arithmetic, row deduplication, logical-to-physical mapping, candidate order,
selected action, random stream, or search allocation. Returned predictions
were scattered back into original request order.

## Exactness

The treatment passed 200 varied exact-service requests on john2 and john3
with zero maximum error and deterministic repeats. An eight-decision R600
rollout-wave smoke passed every candidate, selected-action, rollout-sample,
prediction-error, fallback, and shutdown gate. Its overall report is
intentionally false only because the deliberate eight-decision limit cannot
satisfy the complete canonical-trajectory gate.

Treatment and control diagnostics on both workers reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

After removal, the complete default and `mid-features,v4-opp` workspace
library suites passed. The legacy AI contributed 84 and 85 tests
respectively, legacy core contributed 125 tests in each configuration, and
the focused Python MLX client/service suite passed all 15 tests.

A fresh release build is byte-for-byte identical to the retained
pre-experiment source control, SHA-256
`786351ea84e4b2674e81f2ade87d0596e47a8a3b21be2f336dc9e6ff62c4cd94`.

## Mechanism Diagnostic

The intended row locality was achieved. In the largest retained request,
1,298 rows contained 271,588 feature occurrences:

| Layout | Adjacent trie edges | Reduction |
|---|---:|---:|
| Canonical request order | 258,189 | 4.934% |
| Lexicographic treatment | 174,862 | 35.615% |

That locality did not translate into a sufficient device gain:

| Host | Control MLX eval | Treatment MLX eval | Result |
|---|---:|---:|---:|
| john2 | 7,681.061 ms | 7,681.069 ms | 0.000% slower |
| john3 | 7,505.444 ms | 7,412.169 ms | 1.258% faster |

Sorting complete variable-length rows and scattering outputs dominated the
result before and after MLX. Rust-side neural evaluation increased from
9,282.574 to 10,359.865 ms on john2, an 11.606% regression, and from
9,182.529 to 10,141.388 ms on john3, a 10.442% regression.

## Source-Level Screen

The same matched non-PGO treatment-capable binary was crossed with the switch
off and on. john2 used treatment-control-control-treatment; john3 used the
opposite order. Every run preserved the complete frozen vector.

| Host | Control mean | Treatment mean | Treatment result |
|---|---:|---:|---:|
| john2 | 15.216156 s | 16.233674 s | 6.687% slower |
| john3 | 14.993586 s | 16.052421 s | 7.062% slower |
| Combined | **15.104871 s** | **16.143048 s** | **6.873% slower** |

The combined speedup was `0.93569x`, far below the preregistered requirement
of more than `1.01x`, and both workers regressed. Combined maximum resident
set size was exactly flat at 102,907,904 bytes. Combined allocator peak
footprint fell 0.130%, so memory was not the failure mechanism.

## Verdict

Reject before PGO and remove. The cache-locality premise was measurable, but
host sorting and scatter work overwhelmed a flat-to-small MLX benefit. A fresh
PGO campaign was prohibited by the source gate.

The accepted parent-afterstate PGO champion remains unchanged at 15.018871
seconds, or 9.390x versus the 141.027296-second reference. The 10x threshold
remains 14.102730 seconds, leaving 0.916141 seconds or 6.100%.

Machine-readable evidence:
[`exact-mlx-row-locality-order-rejection-v1.json`](exact-mlx-row-locality-order-rejection-v1.json).

