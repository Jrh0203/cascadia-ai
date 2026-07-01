# Exact Parent Afterstate Feature Context Preregistration

Status: **completed; accepted**

Date: 2026-06-15

Outcome:
`docs/v2/reports/exact-parent-afterstate-feature-context-acceptance-v1.md`.

## Evidence

The accepted seed-34400 K32/R600 stage trace attributes approximately
`1.53` seconds to candidate preparation. Native 1 ms sampling of the accepted
elk-potential PGO binary found 2,932 runnable top-of-stack samples in
`extract_features_with_bag`, versus 293 in the surrounding
`prepare_nnue_move_from_parts` function and 338 in
`BagInfo::from_game_for_player`.

Each rollout policy state prepares up to 15 candidate afterstates. The current
path places one tile and wildlife on a mutable board, then reconstructs every
sparse feature block from the entire board for every candidate. The frozen
game constructs 6,121,807 logical rows.

## Mechanism

The treatment will construct one exact parent feature context per rollout
policy state. It may cache only values that are invariant across that state's
candidate placements and may derive child blocks only from the parent board
plus the already-applied tile and wildlife move.

The treatment must emit the same ordered `Vec<u16>` as
`extract_features_with_bag`, not merely the same feature set. In particular it
must preserve:

- placed-tile insertion order in cell and pair blocks;
- every legacy and v2 pattern feature;
- allowed-wildlife and terrain-pair ordering;
- the historical `mid-features` behavior that emits only in-range prefixes of
  the v3 adjacency block at `[10561,10862)` and does not relocate the extended
  tile-bag blocks;
- the v4 opponent block at `[10862,11231)`;
- the unchanged parent `BagInfo` semantics currently used for candidate
  afterstates.

The general extractor remains the correctness oracle and the fallback for
other feature configurations. The production specialization applies only to
the exact `mid-features,v4-opp` build used by the frozen player.

Candidate generation, candidate order, score arithmetic, search allocation,
row deduplication, MLX requests, random streams, and the benchmark contract
must not change.

## Frozen Contract

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Seed: `34400`
- Four treatment seats
- Candidate budget: K32
- Rollouts: R600 sequential halving
- `MCE_LMR=1`
- `MCE_DIVERSE_PREFILTER=1`
- Full terminal rollouts
- Weights: `nnue_weights_v4opp_modal_iter3.bin`
- Model: `legacy-nnue-v4opp-mlx-v1`

The exact diagnostic vector is:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks.

## Correctness Gates

Before timing, the treatment must:

1. emit byte-identical ordered sparse rows to the full extractor for every
   prepared candidate across at least eight complete seeded four-player AAAAA
   games;
2. cover ordinary drafts, independent drafts, same-cell and different-cell
   wildlife placement, keystones, every tile rotation, and no-wildlife
   fallbacks exercised by the suite;
3. preserve prepared candidate lists, actual scores, selected moves, and
   fallback moves;
4. pass the complete default and `mid-features,v4-opp` library suites;
5. reproduce the frozen score and diagnostic vector on john2 and john3.

Any row-order, candidate, prediction, action, score, or diagnostic mismatch
rejects the treatment before performance measurement.

## Performance Gates

Matched non-PGO release binaries will be crossed on john2 and john3 with two
measurements per binary per host. The treatment advances to fresh race-free
PGO only if:

- combined end-to-end treatment time improves by more than `0.75%`;
- both hosts improve;
- candidate-preparation time falls materially when stage timing is enabled;
- peak memory does not regress;
- every timed run preserves the frozen exact diagnostic vector.

PGO profiles must be collected once per host with `RAYON_NUM_THREADS=1`, then
merged. The final candidate will be crossed against the accepted
elk-potential PGO binary.

## Acceptance

Accept only if the fresh PGO treatment is reproducibly faster on both workers,
remains bit-exact, and improves the accepted result toward the
`14.1027296`-second Phase 0 threshold without an operational regression.
Otherwise remove the specialization and retain a machine-readable rejection
report.
