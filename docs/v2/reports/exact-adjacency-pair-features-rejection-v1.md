# Exact Adjacency Pair Features Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

The qualified legacy NNUE extractor emitted wildlife-pair and terrain-edge
features by converting each placed-tile index to axial coordinates,
constructing three neighbor coordinates, and converting each neighbor back to
an index. The treatment read the same neighbors from the core's precomputed
`ADJACENCY` table.

Direction order, feature order, edge orientation, active model inputs, search
boundaries, and random streams were unchanged.

## Exactness

A focused test compared both pair-feature blocks byte for byte across every
intermediate board in 12 complete seeded four-player AAAAA games.

Every full K32/R600 execution reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches and 6,121,807 logical neural rows;
- 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

## Source-Level Screen

Matched non-PGO release binaries were crossed across john2 and john3.

| Host | Control mean | Treatment mean | Speedup |
|---|---:|---:|---:|
| john2 | 16.178106 s | 15.879138 s | 1.01883x |
| john3 | 15.394750 s | 15.442414 s | 0.99691x |
| Combined | **15.786428 s** | **15.660776 s** | **1.00802x** |

The combined 0.802% source-level gain exceeded the preregistered 0.25%
advancement floor, although the host-level signs disagreed. The experiment
therefore proceeded to fresh PGO rather than being accepted from source timing.

## Fresh PGO

The treatment was instrumented once and trained on one complete R600 game per
worker with `RAYON_NUM_THREADS=1`. This kept LLVM's default counters
race-free. Both profiles contained 5,541 functions and 119,490 blocks. Their
total counts differed by only 62,170 out of roughly 122.09 billion.

The merged profile produced a new LTO release binary. A crossed comparison
against the accepted elk-potential PGO champion then reversed the result:

| Host | Accepted PGO | Treatment PGO | Regression |
|---|---:|---:|---:|
| john2 | 15.341884 s | 15.351917 s | 0.065% |
| john3 | 15.096444 s | 15.170183 s | 0.488% |
| Combined | **15.219164 s** | **15.261050 s** | **0.275%** |

The crossed order was control-treatment-treatment-control on john2 and
treatment-control-control-treatment on john3.

## Verdict

Reject. Precomputed adjacency improved one non-PGO host enough to clear the
source gate, but the production-profiled form regressed on both workers. The
original coordinate loops remain better under the real build and workload.
The helper functions, equivalence oracle, and preregistration were removed; no
runtime switch or experimental path remains.

Machine-readable evidence:
`docs/v2/reports/exact-adjacency-pair-features-rejection-v1.json`.
