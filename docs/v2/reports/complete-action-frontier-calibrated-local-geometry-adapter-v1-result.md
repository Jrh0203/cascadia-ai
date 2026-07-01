# Complete-Action Frontier Calibrated Local-Geometry Adapter V1 Result

Classification: `calibrated_local_geometry_insufficient`.

ADR 0111 isolated exact rotation-canonical local geometry as a zero-initialized residual adapter over the frozen selected model. All four groups used distinct origins and cross-host replay.

## Group Results

| Group | Origin | Replay | Accepted | Completion | Recall | Exact |
|---:|---|---|---:|---|---:|---:|
| 0 | john1 | john4 | 4 | numerically converged | 24.32% | 0.00% |
| 1 | john2 | john1 | 31 | numerically converged | 78.12% | 0.00% |
| 2 | john3 | john2 | 307 | numerically converged | 100.00% | 100.00% |
| 3 | john4 | john1 | 26 | numerically converged | 81.25% | 0.00% |

## Aggregate

- Terminal target recall: 71.13%.
- Terminal exact target sets: 25.00%.
- 120-update aggregate observed: no.

## Gates

| Gate | Result |
|---|---|
| `adapter_pipeline_passed` | pass |
| `all_four_replays_identical` | pass |
| `group_pipeline_passed` | pass |
| `strength_checkpoint_observed` | fail |
| `terminal_strength_gate_passed` | fail |

## Cluster Throughput

- Campaign wall time: 10.97 seconds.
- Scheduled MLX process time: 24.26 seconds.
- Mean active MLX processes: 2.21; peak: 4.
- Idle slot-seconds with compatible queued work: 0.00.
- Duplicate discovery fraction: 0.00%; origins tested distinct groups and all duplication was required cross-host replay.
- Source identity: 116 files, `56e99e0468a161e461de642712fcf6dc7cdfe313b94b609c1e2976a7a8df8628`, identical on john1-john4.

## Decision

The single representation treatment authorized by ADR 0110 is exhausted without meeting the local strength gate. A second representation treatment and full trainer are not authorized.
