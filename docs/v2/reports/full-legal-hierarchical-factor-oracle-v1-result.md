# Full-Legal Hierarchical Factor Oracle V1 Result

Classification: `hierarchical_factor_oracle_sufficient`.

Four distinct static factor-retrieval budgets audited every open train and validation action without training or new teacher compute.

## Results

| Arm | Origin | Replay | Train recall | Validation recall | Validation exact | Mean proposals |
|---|---|---|---:|---:|---:|---:|
| conditional-compact | john1 | john4 | 56.69% | 58.03% | 0.00% | 80.7 |
| conditional-balanced | john2 | john3 | 88.76% | 89.50% | 39.58% | 211.0 |
| conditional-wide | john3 | john2 | 99.27% | 99.18% | 95.00% | 482.4 |
| independent-wide | john4 | john1 | 94.80% | 94.66% | 66.25% | 601.9 |

## Gates

| Gate | Result |
|---|---|
| `all_four_replays_identical` | pass |
| `conditional_wide_strength_passed` | pass |
| `oracle_pipeline_passed` | pass |

## Cluster Throughput

- Campaign wall time: 22.15 seconds.
- Scheduled process time: 61.91 seconds.
- Mean active processes: 2.80; peak: 4.
- Idle slot-seconds with compatible queued work: 0.00.

## Decision

The conditional hierarchy passes the structural Phase 2 gate and authorizes one learned factor-retrieval pilot.
