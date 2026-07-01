# Exact Packed CSR Pipeline Rejection

Status: **rejected and removed**

## Hypothesis

Pipelined exact MLX requests previously crossed the evaluator-thread boundary
as independently allocated sparse rows. The treatment packed each request into
one CSR offset vector plus one contiguous feature vector before submission,
then copied those arrays directly into the existing shared-memory protocol.

The intended gain was fewer row clones, allocations, and row-by-row copy
operations. The model, search budget, row ordering, deduplication, random
streams, and MLX service protocol were unchanged.

## Exactness

All eight full R600 executions reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches and 6,121,807 logical neural rows;
- 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

## End-To-End Measurement

Both hosts ran the sequence treatment, control, treatment, control on frozen
seed 34400. Each number below is the mean of two complete treatment-search
times. The binaries used identical release settings without PGO so the source
change was isolated.

| Host | Control mean | Packed mean | Packed speedup | Regression |
|---|---:|---:|---:|---:|
| john2 | 15.679491 s | 16.216391 s | 0.96689x | 3.424% |
| john3 | 16.005119 s | 16.257209 s | 0.98449x | 1.575% |
| Combined | **15.842305 s** | **16.236800 s** | **0.97570x** | **2.490%** |

Control SHA-256:
`53e62599c9fb9d8c6f232a850d3f7247f668662913297f4ebc36440a21107487`

Treatment SHA-256:
`4d38dacc1abfe16b5d7b394355272ff5d5a78bd205b316176808b66390162058`

## Verdict

Reject. Packing moved the unavoidable feature copy earlier and added it to the
materialization stage. The accepted row-vector path transfers ownership of
already-built rows and copies features once into the shared mapping. The
packed API, trait extension, validation, encoders, and tests were removed.

Machine-readable evidence:
`docs/v2/reports/exact-packed-csr-pipeline-rejection-v1.json`.
