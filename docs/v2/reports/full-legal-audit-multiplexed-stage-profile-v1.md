# Full-Legal Multiplexed Stage Profile

Status: **completed**

Date: 2026-06-15

## Result

The replicated profile is exact on john1, john2, and john3. Every run
reproduced scores `[96,99,92,102]`, terminal state
`7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`,
and normalized semantic BLAKE3
`f46ae73349d53d1baa3c69c0f8a3efab5766ed68ef91b6636ad65a3dea340c75`.

The dominant directly measured critical-path stage is serialized MLX
evaluation:

| Host | Complete wall | Hidden wall | MLX eval | Non-eval MLX | MLX eval share |
|---|---:|---:|---:|---:|---:|
| john1 | 145.693 s | 104.623 s | 49.408 s | 2.802 s | 33.912% |
| john2 | 128.645 s | 90.289 s | 47.982 s | 2.435 s | 37.298% |
| john3 | 128.943 s | 90.793 s | 47.644 s | 2.502 s | 36.950% |
| Remote mean | **128.794 s** | **90.541 s** | **47.813 s** | **2.469 s** | **37.123%** |

Perfectly eliminating MLX evaluation alone would have a maximum remote
end-to-end Amdahl speedup of 1.590x. The 50.321-second mean service interval
leaves 78.473 seconds of wall outside the serialized MLX service.

## Rust Worker Intervals

The Rust counters below are cumulative elapsed intervals from independent
searches. Multiplexed searches and the evaluator worker overlap, so their
905.267-second all-node mean cannot be subtracted from 134.427 seconds of
wall. Percentages intentionally use the serialized stage total, as
preregistered.

| Stage | john1 ms / % | john2 ms / % | john3 ms / % | All-node mean ms / % |
|---|---:|---:|---:|---:|
| Neural evaluation wait | 546,964 / 55.253% | 487,899 / 56.702% | 489,366 / 56.547% | 508,076 / 56.125% |
| Opponent advance | 209,100 / 21.123% | 183,441 / 21.319% | 183,438 / 21.196% | 191,993 / 21.208% |
| Template preparation | 197,681 / 19.969% | 161,293 / 18.745% | 164,330 / 18.989% | 174,435 / 19.269% |
| Rollout initialization | 20,153 / 2.036% | 15,537 / 1.806% | 15,508 / 1.792% | 17,066 / 1.885% |
| Row deduplication | 7,396 / 0.747% | 5,979 / 0.695% | 6,427 / 0.743% | 6,601 / 0.729% |
| Action selection | 2,633 / 0.266% | 2,378 / 0.276% | 2,380 / 0.275% | 2,464 / 0.272% |
| Row materialization | 2,896 / 0.293% | 2,165 / 0.252% | 2,246 / 0.260% | 2,436 / 0.269% |
| Terminal collection | 2,396 / 0.242% | 1,168 / 0.136% | 1,098 / 0.127% | 1,554 / 0.172% |
| Row assembly | 456 / 0.046% | 345 / 0.040% | 353 / 0.041% | 385 / 0.043% |
| Prediction postprocess | 248 / 0.025% | 254 / 0.029% | 270 / 0.031% | 257 / 0.028% |
| Candidate keying | 0.798 / 0.000081% | 0.848 / 0.000099% | 0.790 / 0.000091% | 0.812 / 0.000090% |
| Candidate preparation | 0 / 0% | 0 / 0% | 0 / 0% | 0 / 0% |

On john2 and john3, the mean total was 862.938 cumulative seconds:
56.624% neural wait, 21.258% opponent advancement, and 18.867% template
preparation.

## MLX Workload

Every node served the same 13,437 requests, 44,903,953 rows including the
one-row warmup, and 13,082,538,612 sparse feature occurrences.

| Request rows | Requests | Rows | Row share | MLX eval | Eval share |
|---|---:|---:|---:|---:|---:|
| 1-32 | 120 | 2,185 | 0.005% | 81.430 ms | 0.170% |
| 33-64 | 222 | 9,984 | 0.022% | 134.037 ms | 0.280% |
| 65-128 | 137 | 15,229 | 0.034% | 80.303 ms | 0.168% |
| 129-256 | 406 | 83,264 | 0.185% | 290.410 ms | 0.607% |
| 257-512 | 1,263 | 410,734 | 0.915% | 1,081.274 ms | 2.261% |
| 513-1,024 | 1,203 | 943,686 | 2.102% | 1,701.625 ms | 3.559% |
| Over 1,024 | 10,086 | 43,438,871 | **96.737%** | 44,443.615 ms | **92.954%** |

Remote throughput was 939,176 rows/s and 273.624 million sparse feature
occurrences/s. Launch overhead on small batches is not the main problem.

The largest request had 10,148 rows and 2,185,421 feature occurrences.
Canonical contiguous order exposed only 6.068% full-trie reduction and no
128-feature adjacent prefix. Lexicographic ordering would expose 38.264%, but
prior exact host sorting and prefix-planning experiments already showed that
their bookkeeping erased the isolated kernel gain.

John3's diagnostic-only activation census found H1 65.331% positive and H2
49.828% positive. The dense H1 activation confirms that later-layer sparsity
is not a large escape hatch.

## Ranked Targets

1. **MLX exact neural evaluation.** Direct critical-path wall is 47.813
   seconds, 37.123% of remote wall, with a 1.590x perfect-elimination ceiling.
   The next treatment will change only Metal execution geometry inside the
   existing exact arithmetic order, avoiding host planning or scatter.
2. **Opponent advancement.** It consumes 183.439 cumulative remote worker
   seconds, 21.258% of staged intervals. Normalized by observed 4.533x user-CPU
   concurrency, its capacity-equivalent bound is about 40.467 wall seconds and
   a 1.458x perfect-elimination ceiling.
3. **Rollout-template preparation.** It consumes 162.811 cumulative remote
   worker seconds, 18.867% of staged intervals. The same normalization gives
   about 35.916 wall seconds and a 1.387x perfect-elimination ceiling.

The CPU bounds are ranking estimates, not additive wall attribution, because
CPU preparation overlaps inference and other searches.

## Resource And Correctness Gates

| Host | Maximum RSS | Allocator peak | Process swaps | System swap delta |
|---|---:|---:|---:|---:|
| john1 | 702,660,608 B | 189,907,520 B | 0 | 0 B |
| john2 | 1,038,712,832 B | 146,162,192 B | 0 | 0 B |
| john3 | 1,038,548,992 B | 147,161,592 B | 0 | 0 B |

All maximum RSS values remain below 1.5 GiB. The reports preserved 33,260
logical batches, 55,710,626 logical rows, 44,903,952 physical rows, 29,151
waves, 549,517 samples, zero bootstraps, zero policy fallbacks, zero bridge
fallbacks, and clean shutdown.

The profile binary SHA-256 is
`157ba6c0607f8a2cad3e0ecba9ab6d04cd327175efe01732edb043402d57a5ce`.

Machine-readable evidence:
[`full-legal-audit-multiplexed-stage-profile-v1.json`](full-legal-audit-multiplexed-stage-profile-v1.json).

The complete archive is under
`artifacts/performance/full-legal-audit-multiplexed-stage-profile-v1/`.
