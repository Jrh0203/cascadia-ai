# Exact Bounded Pipeline State Slices Preregistration

Status: **completed - accepted**

Date: 2026-06-15

## Evidence

The accepted direct-template PGO path measures 14.333151 seconds against a
14.102730-second Phase 0 threshold. Only 0.230422 seconds, or 1.608%, remains.

A fresh native sample of the accepted production binary reproduced the frozen
score and diagnostic vector. Candidate generation remains the dominant native
CPU work, but the sample also shows repeated synchronization in
`prepare_next_pipelined_rollout_chunk` and `apply_pipelined_rollout_chunk`.
Source inspection identifies two exact bookkeeping costs around every
96-state pipeline cohort:

1. preparation allocates a Boolean mask sized to the complete rollout-state
   population and scans the complete population in Rayon to select the cohort;
2. application allocates a `usize` position map sized to the complete
   population and scans the complete population in Rayon before advancing only
   the cohort's states through opponent turns.

The active-state vector is collected in strictly increasing state-index order.
Each cohort is a consecutive slice of that vector. Therefore every unfinished
state between the cohort's first and last indices belongs to that cohort; any
gap inside the range is already terminal. The complete-population masks and
scans are unnecessary.

## Mechanism

Add an experimental exact path for the qualified pipelined rollout:

1. derive the half-open state range from the cohort's first and last sorted
   active indices;
2. prepare NNUE moves by scanning only that mutable state slice and selecting
   unfinished states in index order;
3. retain the original global state indices for action selection;
4. advance only the same bounded mutable state slice after applying actions;
5. preserve candidate generation, candidate order, sparse rows, global row
   deduplication, MLX requests, predictions, selected actions, random streams,
   rollout allocation, and terminal scoring exactly.

The treatment must not use unsafe aliasing or unordered indexed mutation. A
same-binary environment switch may select control or treatment during the
source screen. Acceptance removes the switch and obsolete full-population
masking. Rejection removes all treatment code.

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

1. preserve state-index and prepared-group ordering for contiguous and
   terminal-gapped cohorts;
2. match the synchronous pipeline's scores, ordered traces, sparse rows, and
   logical diagnostics;
3. pass the complete default and `mid-features,v4-opp` library suites;
4. pass the focused Python exact-service/client suites;
5. reproduce the frozen score and diagnostic vector on john2 and john3.

Any candidate, row, prediction, selected action, score, sample, fallback,
random-stream, or shutdown mismatch rejects the treatment before performance
measurement.

## Source Performance Gate

Use one matched non-PGO treatment-capable binary on both workers, with two
measurements per mode per host and opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

Advance to production and fresh PGO only if:

- both hosts improve;
- combined end-to-end time improves by more than 0.50%;
- combined template-preparation plus opponent-advance time falls on both
  hosts;
- maximum RSS and peak physical footprint do not materially regress;
- every timed run preserves the frozen exact diagnostic vector.

## Production And PGO Gate

On source-screen success:

1. make bounded slices unconditional in the qualified pipeline;
2. remove the experiment switch, complete-population masks, and dead branch;
3. rerun complete default and `mid-features,v4-opp` library suites plus the
   focused Python exact suites;
4. reproduce source parity on both workers;
5. collect one complete R600 profile per host with
   `RAYON_NUM_THREADS=1`;
6. merge only those two profiles;
7. cross the fresh production PGO binary against the accepted direct-template
   PGO champion in the same opposite balanced ABBA order.

## Acceptance

Accept only if the fresh production PGO binary remains faster on both workers,
is bit-exact, and has no material memory or operational regression. Phase 0
clears only if the crossed accepted time is at or below 14.1027296 seconds,
yielding at least 10.0x end-to-end speedup versus the 141.027296-second frozen
reference.

## Result

Accepted into production. The treatment improved the crossed non-PGO source
binary by 1.004% and the fresh production PGO binary by 1.167%, with positive
results on both john2 and john3. Every measured run preserved the frozen score
and diagnostic vector. Maximum RSS was flat; the allocator footprint
high-water mark rose modestly without changing the operating envelope.

The accepted fleet mean is 14.163055 seconds, or 9.957x versus the frozen
reference. Phase 0 remains open by 0.060325 seconds, or 0.426%.

Full evidence:
[`exact-bounded-pipeline-state-slices-acceptance-v1.md`](exact-bounded-pipeline-state-slices-acceptance-v1.md).
