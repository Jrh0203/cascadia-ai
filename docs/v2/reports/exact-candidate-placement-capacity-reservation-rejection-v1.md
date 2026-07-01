# Exact Candidate Placement Capacity Reservation Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

The exact final length of each market tile's placement vector is known before
its first push:

`frontier.len() * (1 or 6 legal rotations)`.

The treatment called `reserve_exact` once per vector while preserving the
original placement generation order, stable habitat sort, tie behavior, and
top-128 truncation. The intent was to remove repeated geometric growth,
reallocation, and copying from 440,239 rollout-template requests.

## Exactness

A treatment/control oracle compared complete ordered `CandidateMoveSet`
values at every decision in three seeded four-player games, including all-A
and mixed A-D cards, ordinary and forced-Nature-Token boards, overflow
replacement, and dense placement ties. The existing shared-outcome reference
oracle also remained green.

Before timing, the complete default workspace suite passed with 86
`cascadia-ai`, 125 `cascadia-core`, and 61 `cascadia-search` tests. The
`mid-features,v4-opp` suite passed 87 `cascadia-ai` tests, and all 15 focused
Python exact-service/client tests passed.

Every one of the four mechanism diagnostics and eight formal source runs
reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Mechanism

The treatment reduced the registered target stage and retired instructions on
both workers:

| Host | Control template preparation | Treatment | Reduction | Instruction reduction |
|---|---:|---:|---:|---:|
| john2 | 4,457.858 ms | 4,415.140 ms | 0.958% | 0.752% |
| john3 | 4,530.611 ms | 4,442.075 ms | 1.954% | 0.760% |

Mean diagnostic maximum RSS rose 0.050%; mean diagnostic allocator peak
footprint fell 1.392%. The intended local mechanism therefore existed and
qualified for the formal source screen.

## Source Screen

One treatment-capable non-PGO binary, SHA-256
`15de32ed5e6a6fd7e58a095a7078d2d30bea45e3326d85df8ca70e9e99155c55`,
was crossed in the preregistered opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

| Host | Control mean | Treatment mean | Treatment result |
|---|---:|---:|---:|
| john2 | 14.513657 s | 14.650299 s | 0.941% slower |
| john3 | 14.415170 s | 14.346497 s | 0.476% faster |
| Combined | **14.464414 s** | **14.498398 s** | **0.235% slower** |

Mean maximum RSS rose 0.060%. Mean allocator peak footprint rose from
59,593,130 to 61,374,902 bytes, a 2.990% increase. The maximum treatment
footprint observation remained below the maximum control observation, so the
memory evidence was noisy; the timing gate fails independently.

## Verdict

Reject before PGO. Although exact reservation removed measurable work from
template preparation on both machines, the effect did not survive the complete
pipeline. John2 regressed materially and the combined result was slower,
failing both the positive-on-both-hosts and greater-than-0.25% combined-gain
requirements.

The environment switch, treatment release monomorphization, reservation, and
temporary oracle were removed. The complete post-removal workspace suite
passed with 85 `cascadia-ai`, 125 `cascadia-core`, and 61
`cascadia-search` tests; the feature-gated suite passed 86 `cascadia-ai`
tests; all 15 focused Python tests passed.

The accepted bounded-slice PGO champion remains unchanged at
14.16305453125 seconds, or 9.957x versus the 141.027296-second reference.
The Phase 0 gap remains 0.06032493125 seconds.

Machine-readable evidence:
`docs/v2/reports/exact-candidate-placement-capacity-reservation-rejection-v1.json`.

Raw evidence is archived under
`artifacts/performance/exact-candidate-placement-capacity-reservation-v1/`.
