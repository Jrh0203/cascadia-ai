# Exact MLX H1 SIMD Index Broadcast Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Result

Explicitly broadcasting each CSR offset and sparse feature index within a
32-lane SIMD group was exact but effectively neutral. The treatment improved
two Macs and regressed one; its combined gains missed both preregistered
performance gates.

| Host | Control wall | Treatment wall | Change |
|---|---:|---:|---:|
| john1 | 31.704 s | 32.018 s | **+0.989%** |
| john2 | 29.650 s | 29.418 s | **-0.780%** |
| john3 | 29.327 s | 29.081 s | **-0.838%** |
| Mean | **30.227 s** | **30.172 s** | **-0.180%** |

The isolated MLX service timing was similarly small and inconsistent:

| Host | Control MLX | Treatment MLX | Change |
|---|---:|---:|---:|
| john1 | 10,574 ms | 10,704 ms | **+1.235%** |
| john2 | 11,088 ms | 10,993 ms | **-0.859%** |
| john3 | 10,807 ms | 10,541 ms | **-2.463%** |
| Mean | **10,823 ms** | **10,746 ms** | **-0.711%** |

Negative change is faster. The advance gate required at least 1.0% combined
wall improvement and 3.0% combined MLX improvement. Neither passed, so the
full turns-12/39/66 crossover was not authorized.

The result indicates that repeated integer index loads are not a material
bottleneck in this kernel. They are tiny beside the 512 first-layer weights
read for each feature, and the GPU can already coalesce or cache identical
same-address loads. An explicit lane-zero branch and SIMD collective therefore
cannot provide a robust end-to-end gain.

## Exactness

All twelve same-binary crossover runs reproduced:

- scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- normalized turn-66 semantic BLAKE3
  `6f19d82622bab6a5a45c6cdf6e1152f99791630436d1a0354d9f629f95089863`;
- 5,469 logical neural batches;
- 9,288,014 logical and 7,198,144 physical neural rows;
- 4,355 rollout waves and 104,615 samples;
- 13 multiplex cohorts, 104 searches, and 891,486 coalesced rows;
- zero bootstraps, zero policy fallbacks, zero bridge fallbacks, and clean
  shutdown.

Before timing, direct H1 tensors and final outputs matched bit for bit on
empty, duplicate, prefix-related, arbitrary, and 4,096-feature rows. The
treatment-capable tree passed 16 focused Python tests, 25 differential library
tests, 3 trusted fixtures, and 11 NNUE batch tests.

Every process reported zero swaps. Maximum RSS was 557,694,976 bytes and
maximum allocator peak was 184,894,016 bytes, both comfortably below the
1.5-GiB gate. System swap usage was unchanged on all three Macs.

## Removal

The alternate H1 kernel, environment selector, sanitizer allowance, and
treatment-only tests were removed.

The restored Python evaluator returned byte-for-byte to SHA-256
`6556f9c5818354de6847a0630e070f0bb2026252a7bacfa706a094f9f17a9ae9`.
The rebuilt audit binary returned byte-for-byte to the stage-profile binary at
SHA-256
`157ba6c0607f8a2cad3e0ecba9ab6d04cd327175efe01732edb043402d57a5ce`.

The rejected treatment-capable evaluator SHA-256 was
`6354c120ab0710a0607e745aa2c4f1ad781f4c46beb57f6c90ea403f9b60a2ac`,
and its audit binary SHA-256 was
`45978aded0240332afeddca5f6c16b21864c54b12c6b0d32faf3afbef380d2e2`.

The restored tree passed 15 focused Python tests, 25 differential library
tests, 3 trusted fixtures, and 11 NNUE batch tests.

## Decision

Reject and remove. Two different H1 geometry hypotheses have now failed to
produce a stable gain. The next MLX step is a replicated per-layer timing
diagnostic that separates exact H1, H2, and output execution. Any later
intermediate-tensor or kernel-fusion treatment must target the measured layer
cost rather than assuming H1 dominates the service.

Machine-readable evidence:
[`exact-mlx-h1-simd-index-broadcast-rejection-v1.json`](exact-mlx-h1-simd-index-broadcast-rejection-v1.json).

The complete archive is under
`artifacts/performance/exact-mlx-h1-simd-index-broadcast-v1/`.
