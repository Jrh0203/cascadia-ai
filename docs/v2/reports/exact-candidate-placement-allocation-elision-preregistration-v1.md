# Exact Candidate Placement Allocation Elision Preregistration

Status: **completed - rejected at mechanism diagnostics**

Date: 2026-06-15

## Evidence

The accepted bounded-slice PGO path remains 0.06032493125 seconds, or 0.426%,
short of the mandatory 10x Phase 0 threshold.

The frozen R600 workload makes 440,239 exact rollout-template requests.
`candidate_move_set` is the hottest native symbol in fresh accepted-PGO
samples on both workers. For every request it builds one habitat-placement
vector per available market tile by pushing a known number of elements into
an empty `Vec`. Native stacks show `RawVec::grow_one`, `realloc`, and
`memmove` directly beneath this loop.

Each vector's final length is known before its first push:

`frontier.len() * (1 or 6 legal rotations)`.

The same profile also shows the stable placement sort and its allocation path.
Stability is used only to retain the original nested-loop order when habitat
scores tie.

## Hypothesis

Reserving each placement vector's exact known length once will replace
multiple geometric growth reallocations and copies with one allocation.
Replacing the allocating stable sort with a non-allocating unstable sort whose
comparator includes the original frontier-position and rotation rank will
produce the identical total order without sort scratch storage.

Across 440,239 template requests and up to four market tiles per request, the
removed allocator traffic is expected to reduce template-preparation time
enough to clear the remaining Phase 0 gap after fresh PGO.

## Mechanism

Add an experimental exact placement-storage mode inside
`candidate_move_set_impl`:

1. before pushing a market tile's placements, reserve exactly
   `frontier.len() * max_rotation` elements;
2. generate the same placements in the same frontier-major, rotation-minor
   order;
3. rank by descending habitat score;
4. break equal-score ties by ascending
   `frontier_position * max_rotation + rotation`;
5. use `sort_unstable_by` because the explicit key creates a complete,
   deterministic ordering;
6. truncate to the same top 128 placements.

A same-binary environment switch may select the original growth/stable-sort
control or exact-reservation/total-order treatment during the source screen.
Acceptance removes the switch and control branch. Rejection removes the
treatment.

## Exactness Argument

The control's stable sort orders placements by descending habitat score and
retains their original generation order for equal scores. The treatment's
secondary key is exactly that generation order, so both modes define the same
total ordering before truncation.

Reservation changes capacity only. No placement, habitat preview, combination,
candidate, board mutation, shared outcome, wildlife scan, potential call,
tie, sparse row, random stream, or search decision moves across an existing
boundary.

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

1. compare complete ordered placement lists with the original implementation
   across single- and dual-terrain tiles, tied habitat scores, and lists above
   and below the 128-placement cutoff;
2. compare complete `CandidateMoveSet` values at every decision across
   complete seeded four-player games, mixed scoring-card variants, duplicate
   markets, and boards with and without Nature Tokens;
3. preserve the existing shared-outcome-cache reference parity;
4. pass the complete default and `mid-features,v4-opp` library suites;
5. pass the focused Python exact-service/client suites;
6. reproduce the frozen score and diagnostic vector on john2 and john3.

Any placement, tie, candidate, row, prediction, selected action, score,
sample, fallback, random-stream, or shutdown mismatch rejects the treatment
before timing.

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

1. make exact reservation and explicit total-order sorting unconditional;
2. remove the environment switch, control release branch, and dead code;
3. retain test-only control oracles;
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

## Result

The treatment preserved the complete frozen contract, but template-preparation
time regressed 2.733% on john2 and 3.550% on john3. Retired instructions rose
1.746% and 1.780%, respectively. The explicit frontier/rotation tie comparator
cost more than the stable-sort scratch allocation it removed.

The combined treatment was rejected before the formal source screen or PGO.
The unstable total-order sort and experiment switch were removed. Exact vector
capacity reservation remains independently testable under a new
preregistration with the original stable sort unchanged.

Full evidence:
`docs/v2/reports/exact-candidate-placement-allocation-elision-rejection-v1.md`.
