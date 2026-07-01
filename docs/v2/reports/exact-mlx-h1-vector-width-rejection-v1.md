# Exact MLX H1 Vector-Width Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Result

Packing more exact H1 outputs into each Metal thread made the kernel slower on
every Mac. The width-8 and width-16 treatments failed both the complete-wall
and MLX-evaluation gates, so no full-contract crossover was authorized.

| Host | Width 4 wall | Width 8 wall | Width 8 change | Width 16 wall | Width 16 change |
|---|---:|---:|---:|---:|---:|
| john1 | 32.176 s | 32.614 s | **+1.361%** | 33.009 s | **+2.588%** |
| john2 | 29.636 s | 29.740 s | **+0.351%** | 31.343 s | **+5.758%** |
| john3 | 29.298 s | 29.995 s | **+2.379%** | 31.037 s | **+5.936%** |
| Mean | **30.370 s** | **30.783 s** | **+1.360%** | **31.796 s** | **+4.696%** |

The isolated service timing moved in the same direction:

| Host | Width 4 MLX | Width 8 MLX | Width 8 change | Width 16 MLX | Width 16 change |
|---|---:|---:|---:|---:|---:|
| john1 | 10,952 ms | 11,264 ms | **+2.847%** | 12,049 ms | **+10.011%** |
| john2 | 11,025 ms | 11,602 ms | **+5.234%** | 12,873 ms | **+16.770%** |
| john3 | 10,733 ms | 11,388 ms | **+6.103%** | 12,763 ms | **+18.915%** |
| Mean | **10,903 ms** | **11,418 ms** | **+4.720%** | **12,562 ms** | **+15.210%** |

Reducing threads per row did reduce repeated loop control and sparse-index
loads, but two or four live `float4` accumulators per thread increased register
pressure and reduced effective occupancy enough to dominate. Width 4 remains
the correct execution geometry for this network and hardware generation.

## Exactness

All nine Latin-square runs reproduced:

- scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- turn-66 semantic BLAKE3
  `6f19d82622bab6a5a45c6cdf6e1152f99791630436d1a0354d9f629f95089863`;
- 5,469 logical neural batches;
- 9,288,014 logical and 7,198,144 physical neural rows;
- 4,355 rollout waves and 104,615 samples;
- 13 multiplex cohorts, 104 searches, and 891,486 coalesced rows;
- zero bootstraps, zero policy fallbacks, zero bridge fallbacks, and clean
  shutdown.

Bit-exact unit coverage instantiated all three kernels against the Rust-order
reference before timing. The treatment-capable tree passed 18 focused Python
tests, 26 feature-enabled differential tests, and 11 NNUE batch tests.

Every process reported zero swaps. Maximum RSS was 557,727,744 bytes and
maximum allocator peak was 156,877,304 bytes, both comfortably below the
1.5-GiB gate.

## Removal

The width-8 kernel, width-16 kernel, environment selector, sanitizer
allowance, and treatment-only tests were removed.

The restored Python evaluator returned byte-for-byte to SHA-256
`6556f9c5818354de6847a0630e070f0bb2026252a7bacfa706a094f9f17a9ae9`.
The rebuilt restored audit binary returned byte-for-byte to the stage-profile
binary at SHA-256
`157ba6c0607f8a2cad3e0ecba9ab6d04cd327175efe01732edb043402d57a5ce`.

The rejected treatment binary SHA-256 was
`3bba533876ecb383238af8392203fa72a2734cf22f0b689daab802743141d11e`.

## Decision

Reject and remove. The next exact H1 experiment should preserve the proven
single-`float4` accumulator geometry and reduce replicated sparse-index loads
within each SIMD group, avoiding the register-pressure failure measured here.

Machine-readable evidence:
[`exact-mlx-h1-vector-width-rejection-v1.json`](exact-mlx-h1-vector-width-rejection-v1.json).

The complete archive is under
`artifacts/performance/exact-mlx-h1-vector-width-v1/`.
