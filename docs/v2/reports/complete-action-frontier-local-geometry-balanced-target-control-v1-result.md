# Complete-Action Frontier Local-Geometry Balanced-Target Control V1 Result

Classification: `shared_adapter_capacity_insufficient`.

ADR 0113 changed only supervision, replacing expected-rank cross entropy with balanced target-membership BCE.

## Group Results

| Group | Origin | Replay | Accepted | Recall | Exact |
|---:|---|---|---:|---:|---:|
| 0 | john1 | john2 | 6 | 40.54% | 0.00% |
| 1 | john2 | john1 | 3 | 40.62% | 0.00% |
| 2 | john3 | john4 | 34 | 100.00% | 100.00% |
| 3 | john4 | john3 | 13 | 50.00% | 0.00% |

## Aggregate

- Terminal recall: 59.86%.
- Terminal exact sets: 25.00%.

## Gates

| Gate | Result |
|---|---|
| `all_four_replays_identical` | pass |
| `control_pipeline_passed` | pass |
| `group_pipeline_passed` | pass |
| `strength_checkpoint_observed` | fail |
| `terminal_strength_gate_passed` | fail |

## Cluster Throughput

- Campaign wall time: 5.37 seconds.
- Scheduled process time: 15.61 seconds.
- Mean active processes: 2.90; peak: 4.
- Idle slot-seconds with compatible queued work: 0.00.

## Decision

Direct balanced supervision still misses the gate, closing this shared adapter parameterization.
