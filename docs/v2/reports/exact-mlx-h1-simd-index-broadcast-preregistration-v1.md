# Exact MLX H1 SIMD Index Broadcast Preregistration

Status: **closed: rejected and removed**

Date: 2026-06-15

## Question

The replicated full-legal profile spends 47.813 seconds in serialized MLX
evaluation, 37.123% of remote wall. The retained exact H1 kernel assigns 128
threads to each row. Every thread independently loads the same row offsets and
the same ordered sparse feature index before loading its own four weights.

Can loading each offset and feature index once per 32-lane SIMD group, then
broadcasting it to the other lanes, reduce redundant device-memory traffic
without changing any floating-point operation or increasing accumulator
pressure?

## Treatment

Add one experimental H1 kernel with the retained one-`float4`-accumulator
geometry:

- 128 threads per row;
- 256 threads per Metal threadgroup;
- four 32-lane SIMD groups per row;
- SIMD lane zero loads `offsets[row]`, `offsets[row + 1]`, and each
  `indices[position]`;
- `simd_broadcast_first` distributes those integer values to the other 31
  lanes in that SIMD group.

The arithmetic path remains byte-for-byte shaped like the control. Every
hidden output starts from the same bias, visits features in the same CSR
order, performs the same `float4` additions, applies the same ReLU, and enters
the unchanged H2 and output kernels.

The treatment changes no host row, row order, feature order, multiplicity,
weight, bias, output geometry, later-layer kernel, protocol, prediction order,
search state, random stream, rollout budget, or game rule. It adds no planner,
metadata, sort, scatter, cache, or extra accumulator.

The temporary selector is
`CASCADIA_MLX_H1_SIMD_INDEX_BROADCAST=1`. An absent selector is the
same-binary control. Acceptance removes the selector and replaces the control
kernel with the proven implementation. Rejection removes the treatment kernel,
selector, sanitizer allowance, and treatment-only tests.

## Frozen Contract

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Seed: `60999`
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

For the turn-66 source screen, the frozen exact vector is:

- scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- semantic BLAKE3
  `6f19d82622bab6a5a45c6cdf6e1152f99791630436d1a0354d9f629f95089863`;
- 5,469 logical neural batches;
- 9,288,014 logical and 7,198,144 physical neural rows;
- 4,355 rollout waves and 104,615 samples;
- 13 multiplex cohorts, 104 searches, and 891,486 coalesced rows;
- zero bootstraps, zero policy fallbacks, zero bridge fallbacks, and clean
  shutdown.

## Correctness Gates

Before timing:

1. the control and treatment kernels match the Rust-order reference bit for
   bit on empty, duplicate, prefix-related, arbitrary, and maximum-length
   sparse rows;
2. direct H1 tensors match bit for bit, not only final scalar outputs;
3. focused Python MLX tests pass;
4. feature-enabled differential and NNUE batch tests pass;
5. every timed report validates and preserves the frozen score, state,
   semantic digest, logical work vector, zero bootstrap, zero fallback, and
   clean shutdown;
6. every run reports zero process swaps and RSS below 1.5 GiB.

Any compile failure, bit mismatch, action mismatch, score mismatch, diagnostic
mismatch, or resource-gate failure rejects the treatment immediately.

## Three-Node Source Screen

Run the frozen seed-60999 turn-66 audit four times per Mac using the same
treatment-capable binary:

- john1: control, treatment, treatment, control;
- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

Each run includes the full-legal screen, R1200/R4800 confirmation, paid-wipe
diagnostic, multiplexed realized-hidden continuations, Card AAAAA, four
players, no habitat bonuses, and exact K32/R600 search.

The treatment advances only if:

- combined complete wall improves by at least 1.0%;
- combined MLX evaluation time improves by at least 3.0%;
- at least two of three nodes improve in both metrics;
- no node regresses complete wall by more than 1.0%;
- every correctness and resource gate passes.

## Full-Contract Confirmation

If the source gate passes, run an opposite-order ABBA crossover on john2 and
john3 over the complete turns-12/39/66 audit. The treatment is accepted only
if:

- both workers improve complete wall;
- combined complete wall improves by at least 1.0%;
- both workers reduce MLX evaluation time;
- exact semantics and the complete logical work vector remain unchanged;
- zero-swap and memory gates pass.

The switch-free production implementation must then reproduce the exact result
and pass all focused and workspace test suites before the accepted teacher
baseline moves.

## Evidence Location

Source, binaries, commands, checksums, raw reports, validation output, process
timing, and swap measurements will be archived under
`artifacts/performance/exact-mlx-h1-simd-index-broadcast-v1/`.

## Outcome

The treatment remained bit exact in all twelve runs, but improved combined
complete wall by only 0.180% and combined MLX evaluation by only 0.711%.
john2 and john3 improved slightly while john1 regressed in both metrics. The
1% wall and 3% MLX gates failed, so no full-contract confirmation was
authorized.

The alternate kernel, selector, sanitizer allowance, and treatment-only tests
were removed. Full evidence:
[`exact-mlx-h1-simd-index-broadcast-rejection-v1.md`](exact-mlx-h1-simd-index-broadcast-rejection-v1.md).
