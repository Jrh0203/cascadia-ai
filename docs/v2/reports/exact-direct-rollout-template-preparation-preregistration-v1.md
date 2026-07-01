# Exact Direct Rollout Template Preparation Preregistration

Status: **completed - accepted**

Date: 2026-06-15

Result:
[`exact-direct-rollout-template-preparation-acceptance-v1.md`](exact-direct-rollout-template-preparation-acceptance-v1.md).

## Evidence

The accepted parent-afterstate PGO diagnostic spends about 3.61 seconds
building rollout candidate templates, 1.29 seconds preparing their exact
candidate afterstates, and 0.32 seconds constructing and grouping full public
state keys.

A fresh two-worker reuse audit reproduced the complete frozen vector and
observed exactly:

- 440,239 rollout template requests;
- 440,227 unique exact public states;
- only 12 reusable requests;
- 0.002726% exact public-state reuse.

The current qualified batched path nevertheless:

1. constructs a complete `CandidateCacheKey` for every active state;
2. inserts those keys into a per-wave hash map;
3. builds an indirection from state to unique template;
4. calls `candidate_moves_with_base_pub` for each unique state;
5. constructs the same complete key again inside its thread-local
   single-entry cache;
6. clones every newly generated `CandidateMoveSet` into that cache.

Because the outer grouping already presents unique public states and a
rollout board advances monotonically between waves, the inner cache cannot
provide meaningful reuse in this path. The observed 12 saved templates are
too few to repay key construction, hashing, indirection, and cloning.

## Mechanism

Add an experimental direct path used only by the exact batched rollout
pipeline:

1. skip per-wave exact-public-state grouping;
2. generate one template directly for each active rollout state through an
   uncached candidate generator;
3. keep the existing cached candidate API unchanged for scalar and other
   callers;
4. prepare the state-specific candidate rows from that local template;
5. retain candidate order, fallback identity, feature rows, row
   deduplication, MLX requests, selected actions, random streams, and search
   allocation exactly.

The treatment may fuse direct template generation and state-specific
candidate preparation into one Rayon pass so the local template is consumed
without storing or cloning it. It must not use an approximate digest or omit
any candidate work. The experimental path may be gated while measured.
Acceptance removes the gate and obsolete qualified-path grouping. Rejection
removes all treatment code.

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

The exact diagnostic vector is:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Correctness Gates

Before timing, the treatment must:

1. match cached template candidates and fallback identity across complete
   seeded AAAAA games;
2. preserve complete sparse rows, predictions, selected actions, and rollout
   samples in the pipelined parity suite;
3. pass the complete default and `mid-features,v4-opp` library suites;
4. pass the focused Python exact-service/client suites;
5. reproduce the frozen score and diagnostic vector on john2 and john3.

Any candidate, fallback, prediction, action, score, row-count, sample-count,
or random-stream mismatch rejects the treatment before performance
measurement.

## Performance Gates

The same matched non-PGO treatment-capable binary will be crossed on john2
and john3 with two measurements per mode per host. The treatment advances to
fresh race-free PGO only if:

- combined end-to-end time improves by more than `1.00%`;
- both hosts improve;
- combined template preparation, candidate preparation, and candidate-keying
  time falls on both hosts;
- maximum resident set size and allocator peak footprint do not materially
  regress;
- every timed run preserves the frozen exact diagnostic vector.

PGO profiles must be collected once per host with `RAYON_NUM_THREADS=1`, then
merged. The final candidate will be crossed against the accepted
parent-afterstate PGO binary.

## Acceptance

Accept only if the fresh PGO treatment is reproducibly faster on both workers,
remains bit-exact, and improves the accepted 15.018871-second result toward
the 14.102730-second Phase 0 threshold without an operational regression.
Otherwise remove it and retain a machine-readable rejection report.
