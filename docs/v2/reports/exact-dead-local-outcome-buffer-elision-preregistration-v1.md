# Exact Dead Local Outcome Buffer Elision Preregistration

Status: **completed - accepted**

Date: 2026-06-15

## Evidence

The accepted bounded-slice PGO path is 0.06032493125 seconds, or 0.426%,
short of the mandatory 10x Phase 0 threshold.

The frozen R600 workload makes 440,239 direct rollout-template requests.
`candidate_move_set_impl::<true>` uses shared rotation-invariant outcome
caches, but the source still constructs this local buffer before selecting the
shared branch:

```rust
let mut local_outcomes_by_coordinate = vec![None; frontier.len()];
```

Optimized release LLVM IR proves that the allocation survives
monomorphization, fat LTO, and optimization level 3. For every nonempty
combination, the production function:

1. allocates `frontier.len() * 12` bytes at four-byte alignment;
2. initializes every 12-byte `Option<RotationInvariantOutcome>` to `None`;
3. searches or creates the shared outcome cache and uses that shared buffer;
4. frees the unused local allocation at the combination-loop backedge.

When a shared cache entry is absent, a second allocation of the same size is
made for the buffer that is actually used. The first allocation is therefore
not an optimization artifact or source-only concern; it is present in the
accepted machine path.

Every available market produces four combinations without a Nature Token and
sixteen combinations with one. The dead cycle therefore executes at least
four times per nonempty candidate request and up to sixteen times, in the
hottest exact template symbol. This mechanism is distinct from the rejected
placement-vector reservation and metadata-packing experiments.

## Hypothesis

Do not construct the local outcome vector when the const-qualified production
path uses shared outcome caches. Preserve the local vector only for the
test/reference monomorphization that intentionally disables sharing.

Removing millions of small allocations, zero-fill loops, and frees from the
frozen workload should reduce template preparation, allocator pressure, and
retired instructions without changing any search input or output. The saving
may exceed the remaining 0.426% Phase 0 gap after fresh PGO.

## Mechanism

Add two qualified production monomorphizations during measurement:

1. an eager-allocation control that exactly retains the current local
   `vec![None; frontier.len()]` before selecting the shared cache;
2. an elided treatment that represents the unused local vector as an empty
   `Vec` when shared caches are enabled and allocates the full local vector
   only when sharing is disabled.

The shared-cache lookup, cache-entry allocation, cache identity
`(tile_idx, wildlife)`, buffer length, element layout, lookup order, outcome
computation, and cache lifetime remain unchanged.

A same-binary environment switch may select control or treatment for the
source screen. Acceptance removes the switch and eager production
monomorphization while retaining a test-only non-sharing oracle. Rejection
removes the treatment.

## Exactness Argument

In `candidate_move_set_impl::<true>`, the local vector is never selected as
`outcomes_by_coordinate`; only
`shared_outcome_caches[cache_index].outcomes` is read and written. Its
allocation, initialization, and deallocation have no effect on candidate
generation except allocator and memory-system work.

The treatment does not alter combinations, placement vectors, stable sorting,
shared-cache keys, cache hits, computed wildlife outcomes, potential values,
candidate order, fallback identity, board mutation, sparse rows, random
streams, or search budgets. The `REUSE_SHARED_OUTCOMES=false` reference path
continues to allocate and use a full local vector exactly as before.

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

1. compare complete ordered `CandidateMoveSet` values with the eager control
   at every decision across complete seeded four-player games;
2. cover all-A and mixed A-D cards, boards with and without Nature Tokens,
   duplicate wildlife, overflow replacement, and dense habitat ties;
3. preserve the existing shared-outcome-cache versus local-reference parity;
4. prove the local-reference path still allocates a correctly sized local
   outcome vector;
5. pass the complete default and `mid-features,v4-opp` library suites;
6. pass the focused Python exact-service/client suites;
7. reproduce the frozen score and diagnostic vector on john2 and john3.

Any candidate, tie, cache outcome, row, prediction, selected action, score,
sample, fallback, random-stream, or shutdown mismatch rejects the treatment
before timing.

## Mechanism Gate

Run one eager-control and one elided-treatment diagnostic per worker with
complete stage timing, peak-memory accounting, and retired-instruction
measurement.

Advance only if:

- optimized production IR or disassembly contains no unconditional local
  outcome-buffer allocation in the treatment monomorphization;
- the control retains that allocation;
- template-preparation time falls on both workers;
- retired instructions fall on both workers;
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

1. make dead local-buffer elision unconditional for the shared production
   path;
2. remove the environment switch and eager release branch;
3. retain the test-only local-reference path;
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

Accepted. The source screen improved both workers and the combined mean by
1.164%. Production was then rebuilt without the runtime switch from one exact
R600 profile per worker. The fresh PGO binary improved john2 by 1.242%, john3
by 0.781%, and the combined mean by 1.014% while preserving every frozen
score and diagnostic.

The accepted crossed mean is `14.096346521` seconds. This is
`10.004528179689x` faster than the `141.027296`-second reference and
`0.006383079` seconds inside the mandatory 10x threshold. Phase 0 is cleared.

Complete evidence:
`docs/v2/reports/exact-dead-local-outcome-buffer-elision-acceptance-v1.md`.
