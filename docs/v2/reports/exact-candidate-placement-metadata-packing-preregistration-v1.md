# Exact Candidate Placement Metadata Packing Preregistration

Status: **rejected and closed**

Date: 2026-06-15

## Evidence

The accepted bounded-slice PGO path is 0.06032493125 seconds, or 0.426%,
short of the mandatory 10x Phase 0 threshold.

Fresh accepted-PGO native samples identify `candidate_move_set` as the
largest CPU symbol on both workers:

| Host | `candidate_move_set` top samples | `_platform_memmove` top samples |
|---|---:|---:|
| john2 | 8,302 | 1,018 |
| john3 | 8,441 | 1,039 |

The frozen workload prepares 440,239 rollout templates. Each request builds
and stable-sorts one placement vector per available market tile. The current
placement record stores:

- the board index;
- duplicate axial `q` and `r` coordinates;
- the rotation;
- the habitat score.

The coordinates are a pure function of the board index. They are consumed
only when a placement becomes the best candidate or the derived greedy
fallback, but they are currently calculated and copied for every generated
rotation and every stable-sort movement.

The rejected capacity-reservation and unstable-sort experiments do not test
this mechanism. This experiment preserves geometric vector growth and the
existing stable sort exactly; it changes only the element representation.

## Hypothesis

Pack board index and rotation into one `u16` and retain habitat score in a
second `u16`. The transient placement record therefore falls from eight bytes
to four bytes, and eager coordinate conversion disappears from the generation
loop.

Halving element width should reduce allocation volume, stable-sort movement,
cache pressure, and generated instructions inside the hottest exact template
path. Deriving coordinates only for winning placements should cost much less
than storing them for every frontier/rotation pair.

## Mechanism

Add two monomorphized placement representations:

1. the current expanded control record;
2. a packed treatment record containing `packed_index_rotation` and
   `habitat_score`.

For the treatment:

- encode `packed_index_rotation = (board_index << 3) | rotation`;
- decode the board index with `packed >> 3`;
- decode the rotation with `packed & 0b111`;
- derive `HexCoord` from the decoded index only when a placement replaces the
  current best or derived base move;
- preserve the original frontier-major, rotation-minor insertion order;
- preserve the original stable descending-habitat comparator;
- preserve the original top-128 truncation and all later iteration order.

The 21x21 board has indices `0..=440`, and legal rotations are `0..=5`, so the
packed value is injective and fits comfortably in `u16`.

A same-binary environment switch may select expanded or packed placement
records during diagnostics and the source screen. Acceptance removes the
release switch and expanded production branch while retaining a test-only
oracle. Rejection removes the treatment.

## Exactness Argument

Packing is a lossless representation change. It does not alter habitat
preview calls, generated elements, vector growth, insertion order, stable-sort
comparison, tie order, truncation, candidate evaluation, board mutation,
wildlife scoring, potential, sparse rows, random streams, or search budgets.

`HexCoord::from_index` is the inverse already used by the control path before
each record is inserted. Delaying it until a record wins cannot change any
decision because coordinates do not participate in sorting or evaluation.

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

1. prove the expanded record is eight bytes and the packed record is four;
2. round-trip every board index and legal rotation through the packed form;
3. compare complete ordered `CandidateMoveSet` values with the expanded
   control at every decision across complete seeded four-player games;
4. cover all-A and mixed A-D cards, ordinary and forced-Nature-Token boards,
   duplicate wildlife, overflow replacement, and dense habitat ties;
5. preserve the existing shared-outcome-cache reference parity;
6. pass the complete default and `mid-features,v4-opp` library suites;
7. pass the focused Python exact-service/client suites;
8. reproduce the frozen score and diagnostic vector on john2 and john3.

Any candidate, tie, row, prediction, selected action, score, sample, fallback,
random-stream, or shutdown mismatch rejects the treatment before timing.

## Mechanism Gate

Run one expanded and one packed diagnostic per worker with complete stage
timing, peak-memory accounting, and retired-instruction measurement.

Advance only if:

- `template_preparation_ms` falls on both workers;
- retired instructions fall on both workers;
- the packed type is exactly four bytes in the production build;
- maximum RSS and allocator peak footprint do not materially regress;
- all four runs preserve the frozen contract.

## Source Performance Gate

Use one matched non-PGO treatment-capable binary on both workers, with two
measurements per mode per host and opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

Advance to production and fresh PGO only if:

- both hosts improve;
- combined end-to-end time improves by more than 0.25%;
- maximum RSS and peak physical footprint do not materially regress;
- every timed run preserves the frozen exact diagnostic vector.

## Production And PGO Gate

On source-screen success:

1. make the packed representation unconditional for the qualified path;
2. remove the environment switch and expanded release branch;
3. retain a test-only expanded representation oracle;
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

## Outcome

The four-byte record passed every correctness and mechanism gate. Template
preparation fell 1.792% on john2 and 1.075% on john3, and retired
instructions fell on both workers.

The balanced source screen nevertheless regressed john2 by 0.631%, john3 by
0.116%, and the combined mean by 0.375%. Production conversion and PGO were
not authorized. The treatment and temporary tests were removed.

Full evidence:
[`exact-candidate-placement-metadata-packing-rejection-v1.md`](exact-candidate-placement-metadata-packing-rejection-v1.md).
