# Complete-Action Frontier Free-Residual Audit V1 Result

Classification: `free_residual_pipeline_invalid`.

ADR 0103 separated objective geometry, free-parameter optimization, and long-horizon neural local fit on the frozen ADR 0102 cohort. Sealed test, gameplay, new teacher compute, cloud, and external compute remained unused.

## Objective And Optimizer

| Diagnostic | Recall | Exact sets | Mean objective |
|---|---:|---:|---:|
| analytic box optimum | 100.00% | 100.00% | 2.967621 |
| selector ceiling | 100.00% | 100.00% | 4.209584 |
| free AdamW, 120 updates | 39.72% | 0.00% | 4.665333 |
| free AdamW, 1,200 updates | 59.22% | 0.00% | 3.762120 |
| projected control | 96.47% | 79.17% | 2.910821 |

- Analytic maximum KKT violation: `1.471e-15`.
- Projected maximum KKT violation: `3.304e-08` against `1e-8`.
- Projected maximum objective gap: `2.622e-07` against `1e-7`.

## Long-Horizon Neural Fit

| Group | Host | Recall at 120 | Recall at 1,200 | Exact at 1,200 |
|---:|---|---:|---:|---:|
| 0 | john4 | 27.03% | 37.84% | 0.00% |
| 1 | john1 | 56.25% | 84.38% | 0.00% |
| 2 | john2 | 36.59% | 48.78% | 0.00% |
| 3 | john3 | 37.50% | 68.75% | 0.00% |

- Four-group aggregate at 120 exposures: 38.73% recall, 0.00% exact sets.
- Four-group aggregate at 1,200 exposures: 58.45% recall, 0.00% exact sets.

## Frozen Gates

| Gate | Result |
|---|---|
| `analytic_optimum_passed` | pass |
| `free_adam_passed` | fail |
| `neural_at_1200_passed` | fail |
| `neural_at_120_passed` | fail |
| `pipeline_passed` | pass |
| `projected_control_passed` | fail |
| `selector_ceiling_passed` | pass |

## Cross-Host Replays

| Arm | Group | Origin | Replay | Scientific BLAKE3 |
|---|---:|---|---|---|
| analytic-optimum | - | john1 | john2 | `6ecfbee0e5dbac42f8853aefc142a8e641b26554b98f5b7c470e9dd7dd446e75` |
| free-adam | - | john2 | john4 | `b77ba8a385438d7173e209a7c4c9e60a9de6d87968ecaded9145af702c3cef3a` |
| neural-continuation-shard | 0 | john4 | john2 | `b33143b8c9fb670d7f5db1bc7a84a389ed20f82bb964a338579f9c584a700ea2` |
| neural-continuation-shard | 1 | john1 | john3 | `bc6fe283f9893cb2f3accdca20706e80401e5dd950c28287a53e739dc1ee10c1` |
| neural-continuation-shard | 2 | john2 | john4 | `6857f6a0e4a56004886ce9db5236182b0dbf0b06bee870806852c01aa335da22` |
| neural-continuation-shard | 3 | john3 | john4 | `ab77741c5f581389a1fce3ffb624614680b528981b79883bf092feb4c4b045d8` |
| projected-control | - | john3 | john2 | `6c88fa14dc232521aa282d8d2a80abcef694ef329fbc06ee161ff88d37be7935` |

Every origin/replay scientific payload was identical.

## Campaign Throughput

- Origin decision makespan: 241.21 seconds.
- End-to-end origin plus confirmation makespan: 487.70 seconds.
- Scheduled scientific job time: 1205.11 host-seconds.
- Confirmation compute fraction: 49.81%.
- Duplicate discovery fraction: 0.00%; neural continuation used four disjoint group shards.
- Source identity: 109 files, `e3514797f2df154430b11e7749e221be88c361e0d551abbc83c08c7c7a0644da`, identical on john1, john2, john3, john4.

| Host | Jobs | Scheduled seconds |
|---|---:|---:|
| john1 | 2 | 215.24 |
| john2 | 5 | 222.62 |
| john3 | 3 | 414.12 |
| john4 | 4 | 353.13 |

## Authorized Successor

Preregister and replay only the failed numerical-control gate. No model treatment is authorized from this invalid campaign.
