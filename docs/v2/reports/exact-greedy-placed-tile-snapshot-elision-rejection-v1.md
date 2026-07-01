# Exact Greedy Placed-Tile Snapshot Elision Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

The qualified rollout-opponent greedy evaluator cloned `board.placed_tiles`
on every request solely to release an immutable iterator borrow before
hypothetical wildlife scoring mutates and restores other board fields. The
treatment traversed the same insertion-ordered indices directly, copying each
index to a scalar before the mutable scoring call.

The frozen R600 workload makes 1,390,050 qualified greedy requests, so the
treatment was expected to remove the same number of small vector allocations
and copies without changing any decision.

## Exactness

A test-only snapshot oracle matched indexed traversal across 32 evolving turns
for each of three seeded card sets, including mixed A-D cards,
duplicate-wildlife markets, ordinary boards, and forced-Nature-Token variants.
The complete frozen game then matched in every one of the four diagnostic and
eight formal source runs. The complete default and
`mid-features,v4-opp` Rust suites and the focused Python exact-service suites
passed before timing.

Every diagnostic and formal source run reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Mechanism

The paired diagnostic runs showed the intended local work reduction:

| Host | Snapshot control | Indexed treatment | Reduction |
|---|---:|---:|---:|
| john2 opponent advance | 3,400.107 ms | 3,373.852 ms | 0.772% |
| john3 opponent advance | 3,417.880 ms | 3,410.115 ms | 0.227% |

The exact result vector and every downstream diagnostic remained unchanged.

## Source Screen

One treatment-capable non-PGO binary, SHA-256
`7b93898525d7dac2de6acb786f76faefc594d1f52a11f9446bd92312aeeee51e`,
was crossed in opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

| Host | Control mean | Treatment mean | Treatment result |
|---|---:|---:|---:|
| john2 | 14.532296 s | 14.465929 s | 0.457% faster |
| john3 | 14.295060 s | 14.342268 s | 0.330% slower |
| Combined | **14.413678 s** | **14.404098 s** | **0.066% faster** |

Mean maximum RSS increased by 0.008%. Mean allocator peak footprint increased
from 61,321,654 to 63,787,452 bytes, a 4.021% regression.

## Verdict

Reject before PGO. The treatment produced a small, real reduction in its
targeted native stage, but it failed three preregistered source requirements:
john3 regressed, the combined gain missed the 0.25% floor, and allocator peak
footprint materially increased. A fresh PGO trial would violate the registered
decision rule.

The environment switch, dual release monomorphizations, indexed traversal, and
temporary oracle were removed. The accepted bounded-slice PGO champion remains
unchanged at 14.16305453125 seconds, or 9.957x versus the 141.027296-second
reference. The Phase 0 gap remains 0.06032493125 seconds.

Machine-readable evidence:
`docs/v2/reports/exact-greedy-placed-tile-snapshot-elision-rejection-v1.json`.

Raw evidence is archived under
`artifacts/performance/exact-greedy-placed-tile-snapshot-elision-v1/`.
