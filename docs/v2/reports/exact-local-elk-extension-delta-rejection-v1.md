# Exact Local Elk Extension Delta Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

For a qualified AAAAA move that does not place an elk, existing elk line
geometry is unchanged. Only the newly placed tile and a distinct newly
occupied wildlife cell can change whether an existing line has a usable
backward extension. The treatment applied exact before/after deltas for those
local extension cells and retained the complete adjacency scan when an elk
was placed.

Candidate generation, score arithmetic, feature construction, search budgets,
random streams, and MLX inference were unchanged.

## Exactness

The local result matched complete Card-A elk recomputation across six complete
seeded four-player AAAAA games and every exercised tile rotation, no-wildlife
afterstate, and legal wildlife afterstate. The complete default and
`mid-features,v4-opp` library suites each passed 85 tests.

Both frozen K32/R600 worker screens reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches and 6,121,807 logical neural rows;
- 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

All eight crossed timing runs reproduced the same score and diagnostic vector.

## Source-Level Screen

Matched non-PGO release binaries were crossed across john2 and john3.

| Host | Control mean | Treatment mean | Speedup |
|---|---:|---:|---:|
| john2 | 15.693757 s | 15.772879 s | 0.99498x |
| john3 | 16.101481 s | 16.096343 s | 1.00032x |
| Combined | **15.897619 s** | **15.934611 s** | **0.99768x** |

The treatment added 0.036992 seconds per game, a 0.232% regression. It failed
the preregistered requirement to exceed a 0.25% source-level improvement.

## Verdict

Reject before PGO. The local arithmetic is exact, but the retained adjacency
scan is already short and branch-predictable. Virtual before-state handling,
changed-cell collection, and six local directional checks cost more than the
full scan saves at ordinary game depths.

The local delta, its exhaustive test oracle, and the running preregistration
were removed. The accepted complete adjacency implementation remains the only
runtime Card-A elk potential rule.

Machine-readable evidence:
`docs/v2/reports/exact-local-elk-extension-delta-rejection-v1.json`.
