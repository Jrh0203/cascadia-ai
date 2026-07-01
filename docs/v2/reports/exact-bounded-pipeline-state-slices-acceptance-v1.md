# Exact Bounded Pipeline State Slices Acceptance

Status: **accepted**

Date: 2026-06-15

## Change

The qualified exact rollout pipeline no longer allocates a full-population
Boolean mask and scans every rollout state to prepare each 96-state inference
cohort. It also no longer allocates a full-population state-position map and
scans every rollout state before opponent advancement.

Active indices are strictly increasing. Each inference cohort is a consecutive
slice of that ordered vector, so the half-open range from its first through
last state contains exactly its active states plus any already-terminal gaps.
Production now prepares unfinished states and advances opponents only within
that bounded range.

The implementation uses ordinary disjoint mutable slices, retains global state
indices for action application, and contains no unsafe code. The experimental
switch and both full-population branches were removed after acceptance.

## Exactness

The production pipeline preserves candidate order, ordered sparse rows,
predictions, selected actions, rollout traces, random streams, logical and
physical row diagnostics, and terminal scores. Debug assertions enforce
strictly ordered active indices and exact unfinished-state coverage inside
each bounded range.

The complete sequential gates passed before and after production promotion:

- default workspace libraries: `cascadia-ai` 85, `cascadia-core` 125,
  `cascadia-search` 61, and all other workspace libraries;
- `mid-features,v4-opp`: `cascadia-ai` 86, `cascadia-core` 125,
  `cascadia-search` 61, and all other workspace libraries;
- focused Python exact client/service tests: 15 passed;
- formatting and patch-integrity checks.

Every source, parity, profile-training, PGO, and final diagnostic run
reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Mechanism

Matched source diagnostics measured the exact stages targeted by the change:

| Host | Full-population path | Bounded-slice path | Reduction |
|---|---:|---:|---:|
| john2 | 7,899.706260 ms | 7,827.403737 ms | 0.915% |
| john3 | 7,990.618453 ms | 7,838.457042 ms | 1.904% |

These values are the sum of rollout-template preparation and opponent
advancement. The final PGO diagnostic runs attribute 4.480-4.522 seconds to
template preparation and 3.073-3.085 seconds to opponent advancement. MLX
evaluation remains the largest overlapped stage at 8.834-9.038 seconds.

## Source Screen

The same treatment-capable non-PGO binary was crossed with the switch off and
on in opposite balanced orders, with two measurements per mode per host:

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 14.715458 s | 14.534212 s | 1.232% |
| john3 | 14.574453 s | 14.461672 s | 0.774% |
| Combined | **14.644956 s** | **14.497942 s** | **1.004%** |

Mean maximum RSS fell 0.020%. Mean allocator peak footprint rose 0.872%, with
opposite host-level directions, and did not change the operating envelope.

## Fresh PGO

One complete R600 profile was collected on each worker with
`RAYON_NUM_THREADS=1`. Each profile contained 5,557 functions and 120,252
blocks. Their total counts differed by only 37,347 out of roughly 116.17
billion per host. Only those two runtime profiles were merged.

The production profile-use LTO binary was crossed against the accepted
direct-template PGO champion:

| Host | Direct-template PGO | Bounded-slice PGO | Improvement |
|---|---:|---:|---:|
| john2 | 14.531643 s | 14.306317 s | 1.551% |
| john3 | 14.128995 s | 14.019792 s | 0.773% |
| Combined | **14.330319 s** | **14.163055 s** | **1.167%** |

Mean maximum RSS fell 0.099%. The allocator peak-footprint mean rose 5.180%
or 3.29 MB, while the maximum observed treatment footprint exceeded the
maximum control by only 1.52 MB. This noisy high-water metric did not produce
an RSS, shutdown, or operational regression.

## Verdict

Accept. Bounded state slices remove provably unnecessary full-population
allocation and scanning, improve both workers before and after fresh PGO, and
preserve the frozen exact search contract.

The production path is now **9.957x** faster than the 141.027296-second
reference:

- accepted time: `14.16305453125` seconds;
- 10x threshold: `14.1027296` seconds;
- remaining gap: `0.06032493125` seconds, or 0.426%.

The optimization is accepted, but it does not by itself clear Phase 0.

Machine-readable evidence:
`docs/v2/reports/exact-bounded-pipeline-state-slices-acceptance-v1.json`.

The complete local evidence archive is preserved under
`artifacts/performance/exact-bounded-pipeline-state-slices-v1/`.
