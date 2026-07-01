# Exact MLX H1-H2 Threadgroup Fusion Preregistration

Status: **rejected and removed**

Date: 2026-06-15

## Question

The replicated layer profile attributes 76.410% and 76.491% of corrected
layer time to H1 on john2 and john3. The existing kernels materialize a
`rows x 512` H1 tensor in global memory, then read it in a separate H2 launch.
Across the frozen audit, that boundary moves 183,926,591,488 bytes for the H1
write and read.

Can one exact Metal kernel keep H1 in threadgroup memory and immediately
consume it for H2, eliminating the global H1 intermediate and one launch
without changing any per-output floating-point operation order?

## Treatment

Add one temporary fused H1-H2 kernel:

- 256 threads per Metal threadgroup;
- two rows per threadgroup, matching the retained H1 geometry;
- 128 threads per row compute one `float4` H1 vector exactly as today;
- every H1 value starts at the same bias, visits sparse features in the same
  CSR order, performs the same `float32` additions, and applies the same ReLU;
- the 1,024 H1 floats for the two rows are stored in 4 KiB of threadgroup
  memory rather than a global output tensor;
- after a threadgroup-memory barrier, 64 threads per row compute one H2 scalar
  each;
- every H2 output starts at the same bias, visits inputs `0..511` in the same
  order, performs the same `float32` multiply then add with contraction
  disabled, and applies the same ReLU;
- the existing output kernel remains unchanged.

The scalar-per-output H2 mapping changes scheduling only. Each scalar output
retains the exact operation sequence represented by its lane in the existing
`float4` implementation.

The treatment changes no host row, row order, feature order, feature
multiplicity, model weight, bias, output kernel, protocol, prediction order,
search state, random stream, rollout budget, candidate set, or game rule.

The temporary selector is `CASCADIA_MLX_H1_H2_FUSION=1`. An absent selector is
the same-binary control. Acceptance removes the selector and separate H1/H2
production path. Rejection removes the fused kernel, selector, sanitizer
allowance, and treatment-only tests.

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

The frozen turn-66 vector is:

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

1. the fused H2 tensor and final output match the separate kernels bit for bit
   on empty, duplicate, prefix-related, arbitrary, and 4,096-feature rows;
2. direct comparisons cover odd and even row counts, including a batch larger
   than one threadgroup;
3. outputs match the Rust-order reference bit for bit;
4. focused Python MLX tests pass;
5. feature-enabled differential, trusted-fixture, and NNUE batch tests pass;
6. every timed report validates and preserves the frozen score, state,
   semantic digest, logical work vector, zero bootstrap, zero fallback, and
   clean shutdown;
7. every run reports zero process swaps and RSS below 1.5 GiB.

Any compile failure, bit mismatch, action mismatch, score mismatch, diagnostic
mismatch, or resource-gate failure rejects the treatment immediately.

## Three-Node Source Screen

Run the frozen seed-60999 turn-66 audit four times per Mac using one
treatment-capable binary:

- john1: control, treatment, treatment, control;
- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

The treatment advances only if:

- combined complete wall improves by at least 1.0%;
- combined MLX evaluation time improves by at least 3.0%;
- at least two of three nodes improve in both metrics;
- no node regresses complete wall by more than 1.0%;
- every correctness and resource gate passes.

## Full-Contract Confirmation

If the source gate passes, run an opposite-order ABBA crossover on john2 and
john3 over the complete turns-12/39/66 audit. Accept only if:

- both workers improve complete wall;
- combined complete wall improves by at least 1.0%;
- both workers reduce MLX evaluation time;
- exact semantics and the complete logical work vector remain unchanged;
- zero-swap and memory gates pass.

The switch-free production implementation must reproduce the exact result and
pass the focused and workspace test suites before the accepted teacher
baseline moves.

## Evidence Location

Source, binaries, commands, checksums, direct parity fixtures, raw reports,
validation output, process timing, and swap measurements will be archived
under
`artifacts/performance/exact-mlx-h1-h2-threadgroup-fusion-v1/`.

## Outcome

The treatment remained bit exact but failed every performance advance gate.
Across the three-node crossover, complete wall regressed 1.223% and MLX
evaluation regressed 4.193%. All three nodes regressed in both metrics, and
john1 and john3 exceeded the 1.0% per-node wall-regression ceiling. The full
contract confirmation was therefore not authorized.

The fused kernel, selector, sanitizer allowance, and treatment-only tests were
removed. Full evidence and interpretation are recorded in
[`exact-mlx-h1-h2-threadgroup-fusion-rejection-v1.md`](exact-mlx-h1-h2-threadgroup-fusion-rejection-v1.md).
