# Exact Elk Potential Adjacency Acceptance

Status: **accepted**

## Change

Card-A elk potential previously walked axial coordinates forward and backward
from every elk in each line direction. The accepted implementation traverses
the precomputed adjacency table from maximal line starts and stops once a line
reaches four elk, where the qualified potential has no remaining extension
value.

The arithmetic and tie behavior are unchanged:

- only lines of length two or three can contribute;
- the backward extension cell must accept elk;
- each eligible maximal line contributes 20 potential units;
- singles and lines of four or more contribute zero.

The coordinate implementation remains available only as a test oracle. No
runtime experiment switch remains in production.

## Exactness

A deterministic test compared adjacency and coordinate implementations across
24 complete four-player AAAAA games and every intermediate board.

Eight crossed same-binary R600 measurements on john2 and john3 produced one
identical behavioral digest:
`3059e4a2ac96caa7e14599a804d29fb883f49360eb0947209f41ee3319bf5079`.
The digest covers score breakdowns, game records with timing removed,
translation diagnostics, batch diagnostics, gates, status, and shutdown.

Every run reproduced:

- scores `[102, 96, 92, 95]`, mean `96.25`;
- 3,920 neural batches and 6,121,807 logical neural rows;
- 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

## Same-Binary ABBA

The control enabled the coordinate oracle with
`LEGACY_TEACHER_ELK_POTENTIAL_REFERENCE=1`; treatment used adjacency traversal.
The experiment-only switch was removed after measurement.

| Host | Control mean | Treatment mean | Speedup |
|---|---:|---:|---:|
| john2 | 15.702703 s | 15.607584 s | 1.00609x |
| john3 | 16.161772 s | 16.071198 s | 1.00564x |
| Combined | **15.932238 s** | **15.839391 s** | **1.00586x** |

The reproducible source-level gain was 0.092847 seconds, or 0.586%.

## Fresh PGO

The final production form was instrumented and trained on one complete R600
game on each worker. Only the two runtime profiles were merged. A fresh
profile-use LTO build was then recalibrated against the previously accepted
replay-PGO binary:

| Host | Previous PGO | Elk PGO | Improvement |
|---|---:|---:|---:|
| john2 | 15.363099 s | 15.241030 s | 0.122069 s |
| john3 | 15.864978 s | 15.790932 s | 0.074046 s |
| Combined | **15.614038 s** | **15.515981 s** | **0.098057 s** |

The fresh-PGO comparison confirms a 1.00632x end-to-end gain. It does not by
itself clear the frozen 14.102730-second Phase 0 threshold, so the performance
campaign continues.

Machine-readable evidence:
`docs/v2/reports/exact-elk-potential-adjacency-acceptance-v1.json`.
