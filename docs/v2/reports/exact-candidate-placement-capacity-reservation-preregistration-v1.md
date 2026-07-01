# Exact Candidate Placement Capacity Reservation Preregistration

Status: **completed - rejected at source performance gate**

Date: 2026-06-15

Outcome:
[`exact-candidate-placement-capacity-reservation-rejection-v1.md`](exact-candidate-placement-capacity-reservation-rejection-v1.md)

## Evidence

The accepted bounded-slice PGO path remains 0.06032493125 seconds, or 0.426%,
short of the mandatory 10x Phase 0 threshold.

The frozen R600 workload makes 440,239 exact rollout-template requests. For
every request, `candidate_move_set` builds one placement vector per available
market tile. Each vector starts empty even though its exact final length is
known before the first push:

`frontier.len() * (1 or 6 legal rotations)`.

Accepted-PGO native samples show `RawVec::grow_one`, `realloc`, and `memmove`
directly beneath this push loop. The rejected combined allocation-elision
experiment established that replacing the stable sort with an explicit
frontier/rotation tie comparator adds about 1.8% instructions and is not
viable. This successor changes capacity only and leaves the original stable
sort completely unchanged.

## Hypothesis

Calling `reserve_exact` once with the known final length will replace repeated
geometric growth reallocations and copies with one allocation per available
market tile. It changes no element, length, order, comparator, or sort
algorithm.

Across 440,239 requests and up to four placement vectors per request, the
removed allocator traffic may exceed the remaining 0.426% Phase 0 gap after
fresh PGO.

## Mechanism

Add an experimental exact reservation mode inside
`candidate_move_set_impl`:

1. calculate the existing `max_rotation`;
2. reserve exactly `frontier.len() * max_rotation` elements before the first
   push;
3. run the original frontier-major, rotation-minor generation loop unchanged;
4. run the original stable descending-habitat sort unchanged;
5. truncate to the same top 128 placements.

A same-binary environment switch may select empty-vector geometric growth or
exact reservation during the source screen. Acceptance removes the switch and
control branch. Rejection removes the treatment.

## Exactness Argument

`Vec::reserve_exact` changes capacity only. The placement sequence, stable
sort, equal-score order, truncation, candidates, board mutations, sparse rows,
random streams, and search decisions are byte-for-byte unchanged.

## Frozen Contract

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Seed: `34400`
- Four treatment seats
- Candidate budget: K32
- Rollouts: R600 sequential halving
- `MCE_LMR=1`
- `MCE_DIVERSE_PREFILTER=1`
- Full terminal rollouts
- Pipeline chunk states: 96
- Weights: `nnue_weights_v4opp_modal_iter3.bin`
- Model: `legacy-nnue-v4opp-mlx-v1`

Every diagnostic and timed run must reproduce:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Correctness Gates

Before timing, the treatment must:

1. compare complete `CandidateMoveSet` values with the original growth path at
   every decision across complete seeded four-player games, mixed scoring-card
   variants, duplicate markets, and boards with and without Nature Tokens;
2. preserve the existing shared-outcome-cache reference parity;
3. pass the complete default and `mid-features,v4-opp` library suites;
4. pass the focused Python exact-service/client suites;
5. reproduce the frozen score and diagnostic vector on john2 and john3.

Any candidate, tie, row, prediction, selected action, score, sample, fallback,
random-stream, or shutdown mismatch rejects the treatment before timing.

## Source Performance Gate

Use one matched non-PGO treatment-capable binary on both workers, with two
measurements per mode per host and opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

Advance to production and fresh PGO only if:

- both hosts improve;
- combined end-to-end time improves by more than 0.25%;
- template-preparation time falls on both hosts in diagnostic runs;
- maximum RSS and peak physical footprint do not materially regress;
- every timed run preserves the frozen exact diagnostic vector.

## Production And PGO Gate

On source-screen success:

1. make exact capacity reservation unconditional;
2. remove the environment switch and control release branch;
3. retain a test-only geometric-growth oracle;
4. rerun complete default and `mid-features,v4-opp` library suites plus the
   focused Python exact suites;
5. reproduce source parity on both workers;
6. collect one complete R600 profile per host with
   `RAYON_NUM_THREADS=1`;
7. merge only those two profiles;
8. cross the fresh production PGO binary against the accepted bounded-slice
   PGO champion in opposite balanced order.

## Acceptance

Accept only if the fresh production PGO binary is faster on both workers,
preserves the complete frozen contract, has no material operational
regression, and measures at or below 14.1027296 seconds in the crossed mean.
Only that result clears the 10.0x Phase 0 gate versus the 141.027296-second
reference.
