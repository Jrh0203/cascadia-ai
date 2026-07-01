# Exact Candidate Placement Allocation Elision Rejection

Status: **rejected at mechanism diagnostics**

Date: 2026-06-15

## Hypothesis

The treatment combined two exact changes in the hottest
`candidate_move_set` placement-ranking loop:

1. reserve each market tile's known final placement-vector length once;
2. replace the stable habitat sort with `sort_unstable_by`, using the original
   frontier-position and rotation rank as an explicit equal-score tie key.

The goal was to remove repeated vector growth plus stable-sort scratch
allocation while preserving the identical ordered top 128.

## Exactness

Tests compared the complete ordered treatment and control lists below and
above the 128-placement cutoff, with dense habitat ties. Three complete seeded
four-player games compared every `CandidateMoveSet` under all-A and mixed A-D
cards, both as played and with a Nature Token forced onto the acting board.
The existing shared-outcome reference remained green.

The complete default workspace suite passed with 87 `cascadia-ai`, 125
`cascadia-core`, and 61 `cascadia-search` tests. The
`mid-features,v4-opp` suite passed 88 `cascadia-ai` tests, and all 15 focused
Python exact-service/client tests passed.

All four worker diagnostics reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Mechanism Failure

One treatment-capable binary, SHA-256
`ccef531c7ed941c64255f9d75bef126cf8905b4079ae0e0d7be93aefa2ab5376`,
was run once per mode on each worker with complete stage timing.

| Host | Stable control | Reserve + total-order treatment | Regression |
|---|---:|---:|---:|
| john2 template preparation | 4,538.554 ms | 4,662.612 ms | 2.733% |
| john3 template preparation | 4,510.818 ms | 4,670.930 ms | 3.550% |

Retired instructions rose by 18,987,236,629 on john2 (1.746%) and
19,352,127,526 on john3 (1.780%). The secondary frontier/rotation comparison
on frequent habitat ties outweighed the removed sort scratch and vector
growth.

## Verdict

Reject before the formal source screen and PGO. The registered mechanism gate
required template-preparation time to fall on both workers; it rose materially
on both. The unstable total-order sort and combined experiment switch were
removed.

The capacity-reservation half remains independently plausible because native
profiles directly show repeated `RawVec::grow_one`, `realloc`, and `memmove`
under the placement push loop. It is isolated in the successor
`exact-candidate-placement-capacity-reservation-v1` experiment with the
original stable sort and tie behavior unchanged.

Machine-readable evidence:
`docs/v2/reports/exact-candidate-placement-allocation-elision-rejection-v1.json`.

Raw evidence is archived under
`artifacts/performance/exact-candidate-placement-allocation-elision-v1/`.
