# Exact MLX Layer Profile Preregistration

Status: **completed**

Date: 2026-06-15

## Question

The replicated full-legal profile attributes 47.813 seconds of remote wall to
serialized MLX evaluation, but it does not separate the sparse H1 kernel, the
dense H2 kernel, and the scalar output kernel. Two exact H1 geometry
treatments have now failed to produce a stable material gain.

Which exact neural layer actually dominates device time, and is the next
credible target H1-to-H2 intermediate elimination, H2-to-output fusion, or a
different service path?

## Diagnostic

Add diagnostic-only per-layer timing to the exact MLX service:

1. build the unchanged lazy H1, H2, and output graph;
2. evaluate and synchronize H1;
3. evaluate and synchronize H2;
4. evaluate and synchronize the output;
5. immediately evaluate the already-materialized output again to measure
   per-request host and synchronization bookkeeping overhead;
6. return the unchanged output through the unchanged protocol.

The diagnostic records raw H1, H2, output, and materialized re-evaluation
times globally and by the existing request-size buckets. It also records rows
and sparse features covered by layer timing.

The diagnostic selector is `CASCADIA_MLX_LAYER_TIMINGS=1`. It is valid only
with `CASCADIA_MLX_STAGE_TIMINGS=1`; any other value or missing parent timing
mode fails closed. The selector is diagnostic-only and will be removed after
the profile.

## Interpretation

Three evaluation boundaries add synchronization that production does not
have. Therefore:

- raw layer times are reported exactly as measured;
- the already-materialized re-evaluation time estimates per-request timing
  overhead;
- overhead-corrected layer time is
  `max(raw_layer_time - materialized_reevaluation_time, 0)` for each layer;
- corrected shares are used only to rank layers, not claimed as production
  wall;
- the existing one-boundary replicated stage profile remains the authoritative
  production MLX time.

The report will also calculate exact intermediate traffic:

- H1 tensor: `rows * 512 * 4` bytes per write or read;
- H2 tensor: `rows * 64 * 4` bytes per write or read.

Those byte counts bound the global-memory traffic that a correct fusion could
remove but do not assume all traffic reaches DRAM.

## Frozen Contract

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Seed: `60999`
- Audited turns: `12,39,66`
- Realized-hidden turns: `12,39,66`
- Four treatment seats
- Candidate budget: K32
- Search: R600 sequential halving
- Confirmation: R1200 and R4800
- `MCE_LMR=1`
- `MCE_DIVERSE_PREFILTER=1`
- Full terminal rollouts
- Multiplexed realized-hidden continuations
- Pipeline chunk states: 96
- Weights: `nnue_weights_v4opp_modal_iter3.bin`
- Model: `legacy-nnue-v4opp-mlx-v1`

The expected complete exact vector is:

- scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- normalized semantic BLAKE3
  `f46ae73349d53d1baa3c69c0f8a3efab5766ed68ef91b6636ad65a3dea340c75`;
- 33,260 logical neural batches;
- 55,710,626 logical and 44,903,952 physical neural rows;
- 29,151 rollout waves and 549,517 samples;
- zero bootstraps, zero policy fallbacks, zero bridge fallbacks, and clean
  shutdown.

## Replication

Run the complete frozen profile once on john1, john2, and john3 concurrently.
Each run records source and binary checksums, machine identity, process timing,
RSS, allocator peak, and system swap before and after.

The profile is usable only if:

1. all three reports validate and reproduce the complete frozen semantic and
   logical vectors;
2. layer timing covers every exact scalar MLX request and row;
3. every layer total is positive and finite;
4. materialized re-evaluation overhead is reported separately;
5. the same layer ranks first on all three Macs;
6. corrected layer shares differ by no more than 5 percentage points between
   john2 and john3;
7. every process reports zero swaps and RSS remains below 1.5 GiB.

If these gates fail, no fusion treatment is authorized; the instrumentation
must be corrected and rerun.

## Decision Rule

After successful replication, preregister exactly one next treatment:

- H1 dominates: target H1-to-H2 intermediate traffic or measured H1 memory
  access without changing per-output arithmetic order;
- H2 dominates materially: target an exact H1/H2 scheduling or fusion design
  that preserves dense-layer operation order;
- output and H2 launch/intermediate cost is material: target exact H2-output
  fusion;
- no single layer dominates: leave MLX geometry and return to the measured
  opponent-advance, template-preparation, or exact search-work bottlenecks.

No kernel treatment may be selected from intuition alone after this profile.

## Evidence Location

Instrumentation source, binary, launcher, checksums, raw reports, validation
output, timing logs, and the final profile will be archived under
`artifacts/performance/exact-mlx-layer-profile-v1/`.

## Outcome

All replication gates passed. H1 ranked first on every Mac and accounted for
76.410% and 76.491% of corrected layer time on john2 and john3, a difference
of only 0.081 percentage point. Across the complete audit, the H1
intermediate represented 91,963,295,744 bytes per write or read.

The diagnostic selector and synchronization boundaries were removed after the
profile. The production service and audit binary returned byte for byte to
their accepted hashes.

The next treatment is preregistered in
[`exact-mlx-h1-h2-threadgroup-fusion-preregistration-v1.md`](exact-mlx-h1-h2-threadgroup-fusion-preregistration-v1.md).
Full evidence:
[`exact-mlx-layer-profile-v1.md`](exact-mlx-layer-profile-v1.md).
