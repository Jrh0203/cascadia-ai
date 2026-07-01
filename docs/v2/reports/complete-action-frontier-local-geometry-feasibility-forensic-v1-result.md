# Complete-Action Frontier Local-Geometry Feasibility Forensic V1 Result

Classification: `parameterized_fit_or_optimizer_insufficient`.

ADR 0112 used frozen selected-model inference and exact static analysis only. It used no training, gradients, or optimizer updates.

## Group Results

| Group | Origin | Replay | Base recall | Interval ceiling | Mixed classes |
|---:|---|---|---:|---:|---:|
| 0 | john1 | john2 | 24.32% | 100.00% | 0 |
| 1 | john2 | john1 | 46.88% | 100.00% | 0 |
| 2 | john3 | john4 | 24.39% | 100.00% | 0 |
| 3 | john4 | john3 | 28.12% | 100.00% | 0 |

## Aggregate

- Selected-base recall: 30.28%.
- Independent bounded interval ceiling: 100.00% recall and 100.00% exact sets.
- Mixed exact target/non-target feature classes: 0.

## Gates

| Gate | Result |
|---|---|
| `all_four_replays_identical` | pass |
| `forensic_pipeline_passed` | pass |
| `group_pipeline_passed` | pass |
| `independent_interval_ceiling_passed` | pass |
| `no_mixed_exact_feature_classes` | pass |

## Cluster Throughput

- Campaign wall time: 3.07 seconds.
- Scheduled process time: 10.17 seconds.
- Mean active processes: 3.32; peak: 4.
- Idle slot-seconds with compatible queued work: 0.00.

## Decision

The frozen correction range and exact observable rows can represent the target sets. The remaining failure lies in the parameterized shared fit or its optimizer path.
