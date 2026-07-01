# Exact MLX Pipeline Cohort Size Preregistration

Status: **completed - rejected**

Date: 2026-06-15

Result:
[`exact-mlx-pipeline-cohort-size-rejection-v1.md`](exact-mlx-pipeline-cohort-size-rejection-v1.md).

## Evidence

The accepted direct-template PGO path measures 14.333151 seconds against a
14.102730-second Phase 0 threshold. Only 0.230422 seconds, or 1.608%, remains.

Production stage diagnostics reproduce the frozen vector and report:

- 7,709 exact shared-memory MLX requests;
- about 8.99 seconds in Rust-observed neural evaluation;
- 7.35-7.57 seconds in MLX evaluation;
- 341-346 ms in request decode, validation, and MLX input construction;
- 78-82 ms in graph construction;
- 93-98 ms in output materialization and validation;
- 57-58 ms in response writing.

The qualified pipeline currently prepares at most 96 rollout states per
inference cohort. Increasing that exact cohort can reduce request, conversion,
launch, and response overhead while preserving row order, global row
deduplication, predictions, actions, random streams, and search allocation.
Larger cohorts may also reduce CPU/GPU overlap or increase memory, so the
cohort must be measured rather than assumed.

## Frozen Sweep

Use the accepted direct-template PGO binary with the existing
`LEGACY_TEACHER_MLX_PIPELINE_CHUNK_STATES` configuration hook.

Evaluate exactly these treatment sizes:

- 128;
- 160;
- 192;
- 256.

The control remains 96. Run one complete diagnostic measurement per size per
host. john2 uses order `96,128,160,192,256`; john3 uses the reverse order.
No other size may be introduced after results are visible.

Every run uses:

- protocol `cascadia-aaaaa-4p-base-v1`;
- seed 34400;
- four treatment seats;
- K32, R600 sequential halving;
- `MCE_LMR=1`;
- `MCE_DIVERSE_PREFILTER=1`;
- full terminal rollouts;
- `nnue_weights_v4opp_modal_iter3.bin`;
- `legacy-nnue-v4opp-mlx-v1`;
- exact shared-memory transport;
- NNUE and MLX stage timings.

## Correctness Gate

Every sweep and confirmation run must reproduce:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

Any changed row count, prediction, selected action, score, sample count,
fallback, random stream, or shutdown result rejects that size.

## Selection Gate

Select at most one treatment size. It must:

- be faster than 96 on both john2 and john3 in the fixed sweep;
- have the lowest combined treatment time among sizes satisfying that rule;
- reduce exact MLX request count on both hosts;
- not materially regress maximum RSS or peak footprint.

If no size qualifies, reject the experiment without a formal confirmation.

## Confirmation Gate

Cross the selected size against 96 using the same accepted PGO binary, two
measurements per mode per host, and opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

Advance only if both hosts improve and combined end-to-end time improves by
more than 0.50%.

Then make the selected size the production default, remove reliance on the
environment override, rerun the complete default and
`mid-features,v4-opp` library suites plus the focused Python exact suites, and
collect one fresh `RAYON_NUM_THREADS=1` R600 profile per host. Merge only those
two profiles and cross the fresh production PGO binary against the accepted
direct-template PGO champion.

## Acceptance

Accept only if the fresh production PGO binary remains faster on both workers,
preserves every exact diagnostic, and has no material memory or reliability
regression. Phase 0 clears only if the crossed accepted time is at or below
14.1027296 seconds, yielding at least 10.0x end-to-end speedup versus the
141.027296-second frozen reference.
