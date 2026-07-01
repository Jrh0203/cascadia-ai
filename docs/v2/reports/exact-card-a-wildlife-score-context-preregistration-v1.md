# Exact Card A Wildlife Score Context Preregistration

Status: **completed; rejected after source screen**

Date: 2026-06-15

Outcome:
[`exact-card-a-wildlife-score-context-rejection-v1.md`](exact-card-a-wildlife-score-context-rejection-v1.md).

## Evidence

The accepted parent-afterstate PGO native profile identifies wildlife scoring
inside all three remaining CPU-heavy candidate paths:

- `candidate_move_set`, which prepares rollout templates;
- `best_move_without_potential`, which chooses greedy opponent moves;
- `prepare_nnue_move_from_parts`, which computes candidate actual scores.

The current hypothetical-placement helper mutates wildlife occupancy, appends
to a position list, recomputes an entire scoring category from the board, then
restores the board. Candidate preparation also recomputes the drafted category
and Fox score after physically applying every candidate.

In the accepted john2 native sample, `score_wildlife_after_placement` and
`score_wildlife` account for 1,362 and 450 runnable top-of-stack samples,
respectively. The same profile attributes 9,595 samples to
`candidate_move_set`, 2,126 to `best_move_without_potential`, and 2,158 to
`prepare_nnue_move_from_parts`. A single exact parent context can therefore
remove repeated whole-board work from template preparation, opponent
advancement, and candidate preparation at once.

## Mechanism

For an AAAAA board, construct one immutable `CardAScoreContext` containing the
exact current wildlife scores and only the component or adjacency facts needed
to score one hypothetical wildlife placement.

The treatment may:

- update Bear A from the distinct adjacent Bear components and pair count;
- rescore only the Elk A component formed by the new Elk;
- update Salmon A from adjacent component sizes, validity, and local degrees;
- update Hawk A from the isolated-Hawk count and adjacent Hawks;
- update Fox A from per-Fox adjacent-type masks;
- calculate the Fox A score change caused by placing any wildlife type.

It must not mutate the board, inspect hidden state, approximate a score, alter
component tie behavior, or change the general A/B/C/D scoring implementation.
Non-AAAAA games continue to use the existing complete scorer.

The production specialization applies automatically only when all five
wildlife cards are A. It must replace repeated full scoring in the qualified
paths without adding a runtime experiment switch.

Candidate enumeration, candidate order, habitat and potential arithmetic,
Nature Token handling, feature rows, row deduplication, MLX requests, random
streams, search allocation, and the benchmark contract must not change.

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
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Correctness Gates

Before timing, the treatment must:

1. reproduce `score_all_wildlife` exactly when each context is constructed;
2. match `score_wildlife_after_placement` for every wildlife type and every
   legal empty wildlife slot on every intermediate board from at least 16
   complete seeded four-player AAAAA games;
3. match a full post-placement Fox A score for every wildlife type and legal
   empty wildlife slot on the same boards;
4. cover existing and newly placed tiles, disconnected and merged components,
   Bear pair creation and destruction, Elk line ties, valid and branching
   Salmon runs, Hawk isolation changes, Fox type-mask changes, keystones, and
   no-wildlife fallbacks;
5. preserve candidate lists, candidate scores, fallback moves, sparse rows,
   selected actions, score breakdowns, and search diagnostics;
6. pass the complete default and `mid-features,v4-opp` library suites;
7. reproduce the frozen score and diagnostic vector on john2 and john3.

Any score, ordering, prediction, action, row, or diagnostic mismatch rejects
the treatment before performance measurement.

## Performance Gates

Matched non-PGO release binaries will be crossed on john2 and john3 with two
measurements per binary per host. The treatment advances to fresh race-free
PGO only if:

- combined end-to-end treatment time improves by more than `1.00%`;
- both hosts improve;
- aggregate template preparation, opponent advancement, and candidate
  preparation time falls materially with no individual stage regression;
- peak memory does not regress;
- every timed run preserves the frozen exact diagnostic vector.

PGO profiles must be collected once per host with `RAYON_NUM_THREADS=1`, then
merged. The final candidate will be crossed against the accepted
parent-afterstate PGO binary.

## Acceptance

Accept only if the fresh PGO treatment is reproducibly faster on both workers,
remains bit-exact, and improves the accepted 15.018871-second result toward the
14.102730-second Phase 0 threshold without an operational regression.
Otherwise remove the specialization and retain a machine-readable rejection
report.
