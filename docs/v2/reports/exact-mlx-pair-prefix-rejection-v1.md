# Exact MLX Pair-Prefix Rejection

Date: 2026-06-15

Status: **rejected and removed**

## Hypothesis

Many exact sparse NNUE rows share the same first 128 feature indices. A Metal
kernel could accumulate that prefix once for two rows, then continue each row's
tail independently. The arithmetic order within every output row remained
unchanged, so the path was eligible as a strength-neutral Phase 0 optimization.

## Implementation Tested

- Rust grouped rows only after confirming all 128 prefix indices.
- Shared-memory protocol metadata carried row order and prefix lengths.
- One 256-thread Metal threadgroup evaluated two rows.
- The first 128 threads computed the shared H1 prefix once; both 128-thread
  halves then evaluated their original tails.
- Odd and unpaired rows used the same kernel without prefix reuse.
- The build used a fresh two-host R600 profile and PGO.

The experiment passed 200 exact service-parity iterations with zero error. The
full game preserved scores `[102, 96, 92, 95]`, all search diagnostics, and zero
fallbacks.

## Measurements

The captured 1,298-row batch improved from a 1.084229 ms median to 0.859688 ms,
a 1.261x kernel speedup. End-to-end measurements did not preserve that win:

| Host | Pair prefix | Same-binary control | Change |
|---|---:|---:|---:|
| john2 | 15.419411 s | 15.472646 s | 0.053235 s faster (0.34%) |
| john3 | 15.780847 s | 15.771729 s | 0.009118 s slower (0.06%) |

The robust exact-prefix planner and metadata preparation cost about 0.28 seconds
per game. That consumed the Metal saving and produced a cross-host result
indistinguishable from noise. The best treatment remained slower than the
14.102730-second Phase 0 threshold and slower than the previously accepted
15.031628-second replay-PGO result.

Binary SHA-256:
`0c95c1127299118a43236f93c5014217c4ea237ebae2d461785597cd7805410d`

## Reproduction

Treatment:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 CASCADIA_MLX_PAIR_PREFIX=1 \
./legacy-teacher-pair-fresh-pgo exact-mlx-productive-token-compare \
  --server-program ./uv \
  --model-dir artifacts/models/legacy-nnue-v4opp-mlx-v1 \
  --games 1 --first-seed 34400 --rollouts 600 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output pair-fresh-treatment.json
```

Control used the same command and binary without
`CASCADIA_MLX_PAIR_PREFIX=1`.

## Decision

Reject the optimization. Kernel microbenchmarks do not satisfy the Phase 0
gate, and the matched full-game result was neutral. Message type 9, the host
planner, pair metadata, the pair Metal kernel, the environment switch, and
their tests were removed. The accepted shared-memory transport and vectorized
exact kernels remain.
