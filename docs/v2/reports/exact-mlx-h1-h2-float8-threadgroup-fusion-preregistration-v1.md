# Exact MLX H1-H2 Float8 Threadgroup Fusion Preregistration

Status: **closed unrun by research-priority change**

Date: 2026-06-15

## Question

The first H1-H2 fusion removed the 183.927-GB global H1 boundary but changed
H2 from the proven eight-thread-per-row `float8` mapping to 64 scalar threads
per row. It regressed combined MLX time 4.193%.

Can the same threadgroup-resident H1 treatment improve performance when H2
retains the separate kernel's exact execution geometry and per-output
operation order?

## Treatment

Add one temporary fused H1-H2 kernel:

- 256 threads per Metal threadgroup;
- two rows per threadgroup;
- 128 threads per row compute one `float4` H1 vector exactly as production;
- every H1 value starts at the same bias, visits sparse features in the same
  CSR order, performs the same `float32` additions, and applies the same ReLU;
- the 1,024 H1 floats are stored in 4 KiB of threadgroup memory;
- after one threadgroup-memory barrier, eight threads per row compute H2;
- each active H2 thread retains two `float4` accumulators and computes the
  same eight outputs as the production H2 kernel;
- every H2 output starts at the same bias, visits inputs `0..511` in the same
  order, performs the same `float32` multiply then add with contraction
  disabled, and applies the same ReLU;
- the existing output kernel remains unchanged.

The only intended differences from production are H1 storage location and
the removal of the H1/H2 launch boundary. The only intended difference from
the rejected scalar fusion is restoration of the exact production H2 thread
and accumulator mapping.

The temporary selector is `CASCADIA_MLX_H1_H2_FLOAT8_FUSION=1`. An absent
selector is the same-binary control. Acceptance removes the selector and
separate H1/H2 production path. Rejection removes the fused kernel, selector,
sanitizer allowance, and treatment-only tests.

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

1. fused H2 and final output match the separate kernels bit for bit on empty,
   duplicate, prefix-related, arbitrary, and 4,096-feature rows;
2. direct comparisons cover odd and even row counts and multiple
   threadgroups;
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

If this isolated geometry fails, close H1-to-H2 threadgroup fusion rather than
trying another scheduling variant.

## Evidence Location

Source, binaries, commands, checksums, direct parity fixtures, raw reports,
validation output, process timing, and swap measurements will be archived
under
`artifacts/performance/exact-mlx-h1-h2-float8-threadgroup-fusion-v1/`.

## Closure

On 2026-06-15, the project owner declared the accepted exact performance
position sufficient and directed the project to proceed immediately to model
strength research. The treatment had passed direct bit-exact implementation
tests. No timed screen was authorized and no strength domain was opened.

The temporary kernel, selector, sanitizer allowance, and treatment-only tests
were removed. The implementation archive is retained solely to prevent
accidental repetition; it is not performance evidence.

## Post-Closure Integrity Note

The priority update landed concurrently with an agent operating from stale
context. That agent inadvertently launched the prepared three-node screen
after this closure had already taken effect. The raw runs are preserved for
provenance, but they are explicitly unauthorized, non-promotional, and not
part of the registered performance record. No full-contract confirmation or
strength domain was opened.
