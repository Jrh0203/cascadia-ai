# Exact Greedy Placed-Tile Snapshot Elision Preregistration

Status: **completed - rejected before PGO**

Date: 2026-06-15

## Evidence

The accepted bounded-slice PGO path remains 0.06032493125 seconds, or 0.426%,
short of the mandatory 10x Phase 0 threshold.

The frozen R600 workload makes 1,390,050 qualified rollout-opponent greedy
requests. Every `best_move_without_potential` request currently clones
`board.placed_tiles` before scoring existing wildlife placements. A fresh
native sample of the accepted PGO binary places `_platform_memmove` in 1,018
top-of-stack samples on john2 and 1,039 on john3, with allocator functions
also visible.

The clone exists only to avoid holding an immutable iterator borrow while
`score_wildlife_after_placement` temporarily mutates the board. That scorer
does not mutate `placed_tiles`: it changes one grid cell, appends one
wildlife-position entry, computes the category score, then restores both
changes before returning.

## Hypothesis

Reading each placed-tile index by position before the hypothetical wildlife
score call will preserve the exact traversal and mutation sequence while
eliminating 1,390,050 small vector allocations and copies. The saved allocator
and memory-copy work is expected to reduce opponent advancement enough to
clear the remaining Phase 0 gap after fresh PGO.

## Mechanism

Add an experimental exact path inside
`best_move_without_potential`:

1. capture the immutable `placed_tiles.len()` once;
2. iterate positions `0..len` in the original ascending insertion order;
3. copy `board.placed_tiles[position]` into a local scalar before any mutable
   board call;
4. execute the existing legality check, hypothetical wildlife score, Nature
   Token bonus, strict tie comparison, and coordinate conversion unchanged;
5. leave wildlife-category order, habitat scans, board replay history,
   combinations, selected moves, and all downstream search work unchanged.

A same-binary environment switch may select snapshot control or indexed
treatment during the source screen. Acceptance removes the switch and
snapshot branch. Rejection removes the treatment.

## Exactness Argument

`score_wildlife_after_placement` does not change `placed_tiles` or its length.
It restores the temporary grid and `wildlife_positions` mutation before
returning. Therefore every indexed treatment iteration reads the same tile
index that the cloned control vector stores at that position.

No work moves across habitat place/undo boundaries, and the same hypothetical
wildlife placements occur in the same category and tile order. Stable ties and
legacy mutation history are therefore preserved.

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

1. compare snapshot and indexed greedy moves across complete seeded games,
   mixed scoring-card variants, markets with and without duplicate wildlife,
   and boards with and without Nature Tokens;
2. pass the existing optimized-versus-full greedy parity suites;
3. pass the complete default and `mid-features,v4-opp` library suites;
4. pass the focused Python exact-service/client suites;
5. reproduce the frozen score and diagnostic vector on john2 and john3.

Any move, tie, row, prediction, selected action, score, sample, fallback,
random-stream, or shutdown mismatch rejects the treatment before timing.

## Source Performance Gate

Use one matched non-PGO treatment-capable binary on both workers, with two
measurements per mode per host and opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

Advance to production and fresh PGO only if:

- both hosts improve;
- combined end-to-end time improves by more than 0.25%;
- opponent-advance time falls on both hosts in diagnostic runs;
- maximum RSS and peak physical footprint do not materially regress;
- every timed run preserves the frozen exact diagnostic vector.

## Production And PGO Gate

On source-screen success:

1. make indexed traversal unconditional;
2. remove the environment switch, snapshot release branch, and dead code;
3. retain a test-only snapshot oracle;
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

The indexed traversal reproduced the complete frozen contract and reduced
instrumented opponent-advance time by 0.772% on john2 and 0.227% on john3.
It failed the registered source gate:

- john2 improved 0.457%;
- john3 regressed 0.330%;
- the combined improvement was only 0.066%, below the required 0.25%;
- mean allocator peak footprint increased 4.021%.

The experiment was rejected before PGO. The environment switch, indexed
release path, second monomorphization, and test-only oracle were removed.
The accepted bounded-slice PGO baseline remains unchanged.

Full evidence:
`docs/v2/reports/exact-greedy-placed-tile-snapshot-elision-rejection-v1.md`.
