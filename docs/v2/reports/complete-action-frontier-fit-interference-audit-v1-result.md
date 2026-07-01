# Complete-Action Frontier Fit/Interference Audit V1 Result

Classification: `local_optimization_or_representation_insufficient`.

ADR 0102 ran four different open-train diagnostics concurrently on john1-john4. The sealed test, gameplay, new teacher compute, cloud, and external compute remained unused.

## Nested Fit Scaling

| Groups | Recall | Exact sets | Winner retained | Mean objective |
|---:|---:|---:|---:|---:|
| 1 | 18.92% | 0.00% | 100.00% | 3.549371 |
| 4 | 30.28% | 0.00% | 100.00% | 4.933658 |
| 16 | 36.11% | 0.00% | 87.50% | 4.648447 |
| 64 | 36.68% | 0.00% | 90.62% | 4.658670 |

## Capacity Scaling

| Hidden width | Parameters | Recall | Exact sets | Winner retained |
|---:|---:|---:|---:|---:|
| 96 | 1,404,770 | 37.46% | 0.00% | 87.50% |
| 192 | 5,482,178 | 36.93% | 0.00% | 87.50% |
| 288 | 12,232,226 | 38.78% | 0.00% | 84.38% |

## Interference

- Selected-checkpoint gradients opposing the sum of other groups: 78.12%.
- Median cosine to the other-gradient sum: `-0.128186`.
- Off-diagonal pairs at cosine <= -0.10: 44.15%.

| Adaptation | Recall | Exact sets | Winner retained |
|---|---:|---:|---:|
| selected baseline | 32.20% | 0.00% | 79.17% |
| independent per group | 40.66% | 0.00% | 100.00% |
| shared 24-group | 38.43% | 0.00% | 83.33% |

## Frozen Gates

| Gate | Result |
|---|---|
| `capacity_material` | fail |
| `empirical_interference_material` | fail |
| `gradient_interference_material` | pass |
| `local_recovery_passed` | fail |
| `pipeline_passed` | pass |
| `scaling_collapse_material` | fail |

## Execution

| Arm | Host | Seconds | Peak RSS GiB | Process swaps | System swap delta |
|---|---|---:|---:|---:|---:|
| capacity-scaling | john2 | 975.01 | 0.86 | 0 | 0 |
| error-anatomy | john4 | 813.87 | 0.87 | 0 | -8388608 |
| gradient-conflict | john3 | 10.21 | 1.31 | 0 | 0 |
| nested-subset | john4 | 795.28 | 0.86 | 0 | 0 |

- Critical-path arm time: 975.01 seconds.
- Productive arm wall time summed across hosts: 2594.37 host-seconds.
- Frozen diagnostic decisions per critical-path hour: 14.77.
- Duplicate training fraction: 0.0%.
- Source identity: 108 files, `2fc6f314171b8a8f870e1b5a8aa4b93871db140b179108e927440b55c40e049f`, identical on john1, john2, john3, john4.
- Frozen cohort: `30899dec701f053d96023f963b473681516fb0df00a58edf54146c623fd2769d`.

## Cross-Host Replays

| Arm | Origin | Replay | Scientific BLAKE3 | Result |
|---|---|---|---|---|
| capacity-scaling | john2 | john3 | `b431a6e03cbf1f5d3151f604fbff85b7db12d16b921014044e8a30293e4a0098` | identical |
| error-anatomy | john4 | john1 | `78d0de9bd72775d18470336eb68fc7fd9ac47b67b9cadc5ed5c3ec880c7e2949` | identical |
| gradient-conflict | john3 | john4 | `c19ec1dfe57fbc938953b1d927dbed1ccabbedb55d7e31fe192379200ec3527a` | identical |
| nested-subset | john1 | john4 | `53393670ce1d1a961303f3ecb73153b2c5253fa34d5c61a95e8177c3e39c1c19` | identical |

The john1 nested origin recorded zero process swaps but unrelated positive system-wide swap growth. Its john4 replay was scientifically identical with zero swap growth and is the pipeline-selected nested report.

## Campaign Throughput

- First-wave decision makespan: 975.30 seconds.
- End-to-end origin plus confirmation makespan: 1961.49 seconds.
- Scheduled scientific job time: 5260.36 host-seconds.
- Confirmation compute fraction: 50.19%.
- Duplicate discovery fraction: 0.00%; all repeated compute was the preregistered cross-host confirmation wave.
- The campaign was MLX-bound, so the plan's CPU-bound 85% physical-core target does not apply to this diagnostic.

| Host | Jobs | Scheduled seconds |
|---|---:|---:|
| john1 | 2 | 1675.25 |
| john2 | 1 | 975.10 |
| john3 | 2 | 990.31 |
| john4 | 3 | 1619.70 |

## Authorized Successor

Test one bounded representation or local-optimizer mechanism. A larger shared model or conflict-only treatment is not yet authorized.
