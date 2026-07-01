# Exact MLX Layer Profile

Status: **completed and instrumentation removed**

Date: 2026-06-15

## Result

H1 is the dominant exact MLX layer on all three Macs. The independent remote
nodes agree almost perfectly:

| Host | Complete wall | H1 corrected | H1 share | H2 corrected | H2 share | Output corrected | Output share |
|---|---:|---:|---:|---:|---:|---:|---:|
| john1 | 149.927 s | 43.469 s | 78.900% | 8.184 s | 14.855% | 3.441 s | 6.245% |
| john2 | 136.155 s | 43.173 s | 76.410% | 9.477 s | 16.773% | 3.851 s | 6.816% |
| john3 | 135.642 s | 42.687 s | 76.491% | 9.366 s | 16.783% | 3.753 s | 6.726% |
| Remote mean | **135.898 s** | **42.930 s** | **76.451%** | **9.422 s** | **16.778%** | **3.802 s** | **6.771%** |

The john2/john3 H1-share difference is 0.081 percentage point, far inside the
5-point replication gate. Materialized-output re-evaluation overhead averaged
only 7.654 milliseconds over 13,437 remote requests.

The diagnostic inserted three synchronization boundaries, so these layer
times rank work but are not production wall attribution. The authoritative
one-boundary production MLX time remains 47.813 seconds. Applying the
replicated layer share to that interval estimates 36.553 seconds in H1 and a
1.396x end-to-end perfect-H1-elimination ceiling. A fusion can remove only
part of that cost.

## Intermediate Traffic

Every node timed 13,437 exact requests, 44,903,953 rows including warmup, and
13,082,538,612 sparse feature occurrences.

| Intermediate | Bytes per write or read | Write plus read |
|---|---:|---:|
| H1, `rows x 512 x f32` | 91,963,295,744 | 183,926,591,488 |
| H2, `rows x 64 x f32` | 11,495,411,968 | 22,990,823,936 |

The first fusion target is therefore the H1-to-H2 boundary. Keeping each
512-float H1 row in Metal threadgroup memory can remove its global write and
read plus one kernel launch while preserving every sparse feature addition
and H2 input accumulation in order.

## Exactness And Resources

All three reports validated and reproduced:

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

Layer timing covered every exact request and row, and every layer total was
positive and finite. H1 ranked first on all nodes.

| Host | Maximum RSS | Allocator peak | Process swaps | System swap delta |
|---|---:|---:|---:|---:|
| john1 | 969,097,216 B | 174,391,896 B | 0 | 0 B |
| john2 | 1,117,896,704 B | 157,205,056 B | 0 | 0 B |
| john3 | 1,118,322,688 B | 149,733,904 B | 0 | 0 B |

All RSS observations remain below 1.5 GiB.

## Removal

The layer selector, forced synchronization boundaries, counters, sanitizer
allowance, and temporary tests were removed.

The restored service is byte-identical to SHA-256
`3e9d62ee5d1f38b0b2ae1fe8ac32d312b3c1dcb25b4fa5b4cc668b7212b457a1`.
The rebuilt audit executable is byte-identical to the accepted production
binary at SHA-256
`157ba6c0607f8a2cad3e0ecba9ab6d04cd327175efe01732edb043402d57a5ce`.

The diagnostic tree passed 17 focused Python tests, 25 differential library
tests, 3 trusted fixtures, and 11 NNUE batch tests. The restored tree passed
15 focused Python tests and the same 25, 3, and 11 Rust tests.

## Decision

Preregister one exact threadgroup-resident H1-to-H2 fusion. The treatment
keeps the current 128-thread-per-row H1 arithmetic, stores H1 in threadgroup
memory, and computes H2 in the same kernel without a global H1 tensor. It must
pass direct bit-exact H2/output comparison before any timed screen.

Machine-readable evidence:
[`exact-mlx-layer-profile-v1.json`](exact-mlx-layer-profile-v1.json).

The complete archive is under
`artifacts/performance/exact-mlx-layer-profile-v1/`.
