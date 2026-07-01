# Exact MLX Pipeline Cohort Size Rejection

Status: **rejected**

Date: 2026-06-15

## Experiment

The accepted exact pipeline uses 96 rollout states per inference cohort. A
preregistered two-host sweep tested only 128, 160, 192, and 256, with opposite
host order and complete NNUE/MLX stage diagnostics.

Every run reproduced the frozen score and search vector exactly:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Result

| Cohort states | john2 | john3 | Combined | Result vs 96 |
|---:|---:|---:|---:|---:|
| 96 | 14.383086 s | 14.138662 s | **14.260874 s** | control |
| 128 | 15.034221 s | 14.942792 s | 14.988507 s | 5.102% slower |
| 160 | 15.072125 s | 14.829248 s | 14.950687 s | 4.837% slower |
| 192 | 14.994922 s | 14.894456 s | 14.944689 s | 4.795% slower |
| 256 | 15.006548 s | 14.879722 s | 14.943135 s | 4.784% slower |

All four treatments regressed both hosts, so none passed the preregistered
selection gate and no formal confirmation or fresh PGO build was authorized.

## Mechanism

The larger cohorts did reduce service work:

- requests fell from 7,709 to 4,113 at 128 and 3,957 at 160-256;
- mean decode/validation fell from 339.7 ms to 157-167 ms;
- mean graph construction fell from 77.8 ms to 22.8-25.4 ms;
- mean MLX evaluation fell from 7,405 ms to 6,404-6,523 ms.

Those isolated sums are overlapped in the 96-state pipeline. Larger cohorts
lengthened each producer/consumer interval, reduced pipeline overlap, and
increased working-set size. Mean maximum RSS rose from 115.0 MB at 96 to
150.7 MB at 128 and 191.3 MB at 256. The reduced request overhead therefore
did not translate into end-to-end throughput.

## Verdict

Reject. Keep the 96-state production default. No code or production
configuration changed.

Machine-readable evidence:
`docs/v2/reports/exact-mlx-pipeline-cohort-size-rejection-v1.json`.

The complete evidence archive is preserved under
`artifacts/performance/exact-mlx-pipeline-cohort-size-rejection-v1/`.
