# Exact Incremental Top-Eight Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

The qualified greedy policy retained its stable top-eight habitat placements
by repeatedly finding the worst selected element and sorting the final eight.
The treatment instead kept the array in descending score order while scanning,
inserting equal scores after earlier placements.

The scan, habitat previews, traversal order, candidates, tie rules, random
streams, and search contract were unchanged.

## Exactness

Unit tests compared the bounded top-eight result with a stable full sort,
compared optimized and reference greedy moves with and without Nature Tokens,
and replayed 16 complete four-player AAAAA games.

Every full R600 execution reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches and 6,121,807 logical neural rows;
- 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

## Source-Level Screen

Both hosts ran treatment, control, treatment, control with matched non-PGO
release binaries.

| Host | Control mean | Treatment mean | Speedup |
|---|---:|---:|---:|
| john2 | 15.701642 s | 15.591887 s | 1.00704x |
| john3 | 16.113529 s | 16.049694 s | 1.00398x |
| Combined | **15.907585 s** | **15.820791 s** | **1.00549x** |

The 0.549% source-level gain exceeded the preregistered 0.25% advancement
floor, so the experiment proceeded to fresh PGO.

## PGO Audit

The first fresh profile was collected with LLVM's default non-atomic
instrumentation while Rayon executed the workload on ten workers. Identical
gameplay produced materially different hot-block counts because concurrent
counter increments were lost. Its profile-use build regressed by 4.272% and
was discarded.

Atomic updates for every profiling counter caused severe cache-line
contention: both workers consumed about ten CPU cores but had not reached the
tenth decision after more than three minutes. Those instrumented-only runs
were stopped.

The final profile collection used one Rayon worker per host, making the
default counters race-free. The two profiles each contained 5,537 functions
and 119,172 blocks. Their total counts differed by only 1,436 out of
119,470,389,930. The profiles were merged, then the resulting production
binary was benchmarked with the real ten-core workload.

| Host | Accepted PGO | Treatment PGO | Regression |
|---|---:|---:|---:|
| john2 | 15.419406 s | 15.542713 s | 0.800% |
| john3 | 15.848589 s | 15.912847 s | 0.405% |
| Combined | **15.633998 s** | **15.727780 s** | **0.600%** |

The crossed order was control-treatment-treatment-control on john2 and
treatment-control-control-treatment on john3.

## Verdict

Reject. The insertion-maintained array was faster without PGO, but the
existing worst-element scan and final eight-element sort are better after
production profile-guided optimization. The treatment code and
preregistration were removed.

Machine-readable evidence:
`docs/v2/reports/exact-incremental-top8-rejection-v1.json`.
