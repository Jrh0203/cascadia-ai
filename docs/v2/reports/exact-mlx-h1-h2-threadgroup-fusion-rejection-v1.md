# Exact MLX H1-H2 Threadgroup Fusion Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Result

Keeping H1 in threadgroup memory was exact, but the scalar H2 consumer made
the complete evaluator slower on every Mac.

| Host | Control wall | Treatment wall | Change |
|---|---:|---:|---:|
| john1 | 31.695 s | 32.161 s | **+1.472%** |
| john2 | 29.675 s | 29.894 s | **+0.738%** |
| john3 | 29.165 s | 29.587 s | **+1.446%** |
| Mean | **30.178 s** | **30.547 s** | **+1.223%** |

The isolated MLX service timing regressed more:

| Host | Control MLX | Treatment MLX | Change |
|---|---:|---:|---:|
| john1 | 10,581 ms | 11,103 ms | **+4.936%** |
| john2 | 11,156 ms | 11,577 ms | **+3.779%** |
| john3 | 10,731 ms | 11,149 ms | **+3.891%** |
| Mean | **10,823 ms** | **11,277 ms** | **+4.193%** |

Negative change is faster. The treatment failed the 1.0% combined-wall and
3.0% combined-MLX improvement gates, improved zero nodes in both metrics, and
exceeded the 1.0% node-wall regression ceiling on john1 and john3. The full
turns-12/39/66 confirmation was not authorized.

The eliminated H1 global write/read and kernel launch did not compensate for
the changed H2 execution geometry. The retained separate H2 kernel computes
eight outputs per thread with two `float4` accumulators and eight threads per
row. This treatment instead activated 64 scalar H2 threads per row after the
barrier. The consistent three-node regression indicates that the extra
scheduling, threadgroup residency, and barrier cost outweighed the removed
intermediate traffic in this geometry.

## Exactness

Direct real-model parity passed independently on john1, john2, and john3 for
1, 2, 3, and 17 rows, including empty rows, duplicate features, arbitrary
rows, and a 4,096-feature row. Fused H2 tensors and final outputs matched the
separate kernels bit for bit.

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

The treatment-capable tree passed 17 focused Python tests, 25 differential
library tests, 3 trusted fixtures, and 11 NNUE batch tests.

Every process reported zero swaps. Maximum RSS was 557,203,456 bytes and
maximum allocator peak was 239,338,048 bytes, both below the 1.5-GiB gate.
System swap usage was unchanged on all three Macs.

## Removal

The fused kernel, environment selector, sanitizer allowance, and
treatment-only tests were removed.

The restored Python evaluator returned byte-for-byte to SHA-256
`6556f9c5818354de6847a0630e070f0bb2026252a7bacfa706a094f9f17a9ae9`.
The rebuilt audit binary returned byte-for-byte to the accepted production
binary at SHA-256
`157ba6c0607f8a2cad3e0ecba9ab6d04cd327175efe01732edb043402d57a5ce`.
The restored Rust evaluator boundary returned to SHA-256
`6767a2f7aeaa574b7d16b815d12e3d7d83cec1852d3667165deabf40a0a0b921`.

The rejected treatment-capable evaluator SHA-256 was
`a7796da301320b0208bcd8b2d151c77b2883b1c571cdfebc7fbbc1ec2cd0bdac`,
and its audit binary SHA-256 was
`fcbdcd0c480746d2145dd32e806f34ec24e70e819f8c18c9fb41f9040e4f0a51`.

The restored tree passed 15 focused Python tests, 25 differential library
tests, 3 trusted fixtures, and 11 NNUE batch tests. Rust formatting and Python
lint also pass.

## Decision

Reject and remove this geometry. One narrowly distinct fusion remains
justified: retain the proven H2 mapping exactly inside the fused kernel, with
eight H2 threads per row and two `float4` accumulators per active thread.
That isolates H1-intermediate elimination from the scalar-H2 scheduling
change. If it also fails, close threadgroup H1-to-H2 fusion and return to the
measured CPU stages or exact search-work reduction.

Machine-readable evidence:
[`exact-mlx-h1-h2-threadgroup-fusion-rejection-v1.json`](exact-mlx-h1-h2-threadgroup-fusion-rejection-v1.json).

The complete archive is under
`artifacts/performance/exact-mlx-h1-h2-threadgroup-fusion-v1/`.
