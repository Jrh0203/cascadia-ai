# Exact MLX Group-of-Four Prefix Rejection

Status: **rejected and removed**

## Hypothesis

Many sparse NNUE rows in the same rollout cohort share their first 128
features. The experiment grouped matching rows in fours, accumulated the common
H1 contribution once in Metal threadgroup memory, and evaluated only each
row's suffix independently.

The captured 1,298-row batch looked promising:

| Measurement | Baseline | Group-of-four |
|---|---:|---:|
| H1 kernel median | 1.032 ms | 0.799 ms |
| Kernel speedup | - | 1.29x |
| Feature work removed | - | 16.5% |

## Exactness

Both john2 and john3 passed 200-iteration service parity with zero absolute
error and deterministic repeats. Every full R600 measurement reproduced:

- scores `[102, 96, 92, 95]`, mean `96.25`;
- 3,920 neural batches and 6,121,807 logical neural rows;
- 5,062,305 physical rows after exact deduplication;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean service shutdown.

The rollout-wave fixture's R32 trajectory is stale relative to the current
native implementation: current native and MLX produced the same fixture
mismatch. MLX repeated exactly, and all R600 spot checks had zero error.

## Fair PGO Test

The treatment was profile-trained from fresh full R600 runs on john2 and john3,
then rebuilt with the merged runtime-only profile. The final comparison used an
opposite-order ABBA design on the two hosts:

| Host | Control mean | Treatment mean | Treatment delta |
|---|---:|---:|---:|
| john2 | 15.4099 s | 15.4881 s | +0.0782 s |
| john3 | 15.8513 s | 15.8132 s | -0.0381 s |
| Combined | **15.6306 s** | **15.6506 s** | **+0.0200 s** |

The combined speedup was `0.99872x`, or a `0.128%` regression. This is noise-level
neutral and does not reduce the remaining Phase 0 gap.

## Verdict

Reject. The isolated kernel win was consumed by exact prefix discovery,
group planning, metadata transport, output scattering, and the larger
512-thread launch. The complete message-9 protocol, planner, prefix
bookkeeping, Metal kernel, service parser, and tests were removed. The accepted
message-8 shared CSR path remains the production baseline.

Detailed machine-readable evidence:
`docs/v2/reports/exact-mlx-group4-prefix-rejection-v1.json`.
