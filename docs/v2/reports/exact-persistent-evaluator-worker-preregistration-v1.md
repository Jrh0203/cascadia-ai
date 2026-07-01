# Exact Persistent Evaluator Worker Preregistration

Status: **rejected**

Date: 2026-06-15

Result:
`docs/v2/reports/exact-persistent-evaluator-worker-rejection-v1.md`.

The source screen passed, but the fresh production-PGO validation regressed
john3 by 0.714% and measured 59.847 ms above the absolute 10x threshold. The
treatment and all experiment-only production/test paths were removed.

## Evidence

The accepted bounded-slice PGO path is 0.06032493125 seconds, or 0.426%,
short of the mandatory 10x Phase 0 threshold.

The qualified exact pipeline currently creates two bounded channels and spawns
one scoped evaluator thread inside every sequential-halving rollout batch.
A K32 search normally has five halving rounds, so the evaluator process and
its MLX model remain persistent while the Rust thread and channels that drive
them are repeatedly created and joined.

Fresh accepted-PGO native samples captured 104 process threads on john2 and
107 on john3 during roughly 5.8-second windows. Each sample contained the main
thread, ten long-lived Rayon workers, and more than ninety short-lived
evaluator-worker thread identities. Those workers repeatedly alternated
between the model-service read and the request-channel receive.

## Hypothesis

Keeping one evaluator worker and one request/response channel pair alive for
all halving rounds in a single search decision will remove hundreds of thread
and channel lifecycle operations from a complete game. The request contents,
request order, response order, pipeline cohort size, CPU/MLX overlap, and
halving algorithm remain unchanged.

The eliminated setup, teardown, stack allocation, scheduler registration, and
channel allocation may exceed the remaining 0.426% Phase 0 gap after fresh
PGO.

## Mechanism

Add an experimental exact worker-lifetime mode:

1. compute root priors on the calling thread exactly as today;
2. create the existing capacity-one request and response channels once;
3. spawn the existing evaluator loop once for the complete sequential-halving
   search;
4. run every halving round through the same worker and channels;
5. close the request channel and join the worker after the final round or
   immediately after any error;
6. retain the existing per-round worker lifecycle as the control and test
   oracle.

No request may be merged, split, reordered, prefetched across a response
validation boundary, or evaluated concurrently with another request.

A same-binary environment switch may select per-round or per-search lifetime
during the source screen. Acceptance removes the switch and per-round
production branch while retaining a test-only oracle. Rejection removes the
treatment.

## Exactness Argument

The evaluator remains uniquely mutably owned by one scoped thread. The
treatment changes only the lifetime of that thread and its channels. Every
round still initializes the same rollout states, prepares the same ordered
cohorts, sends the same sparse rows, validates the same response before action
selection, advances the same opponents, and joins the next halving round only
after the current batch completes.

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

1. match the per-round oracle's complete ordered evaluator-request stream;
2. match estimates, sample counts, rollout-value samples, sparse-row
   diagnostics, selected actions, and terminal scores bit for bit;
3. preserve evaluator-error, invalid-width, non-finite-output, disconnect, and
   panic handling without deadlock;
4. pass the complete default and `mid-features,v4-opp` library suites;
5. pass the focused Python exact-service/client suites;
6. reproduce the frozen score and diagnostic vector on john2 and john3.

Any request, row, prediction, selected action, score, sample, fallback,
random-stream, error, or shutdown mismatch rejects the treatment before
performance measurement.

## Source Performance Gate

Use one matched non-PGO treatment-capable binary on both workers, with two
measurements per mode per host and opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

Advance to production and fresh PGO only if:

- both hosts improve;
- combined end-to-end time improves by more than 0.25%;
- a treatment native sample materially reduces transient evaluator-worker
  identities relative to the accepted sample;
- maximum RSS and peak physical footprint do not materially regress;
- every timed run preserves the frozen exact diagnostic vector.

## Production And PGO Gate

On source-screen success:

1. make one evaluator worker per search unconditional for the qualified path;
2. remove the environment switch and per-round production branch;
3. retain a test-only per-round lifecycle oracle;
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
