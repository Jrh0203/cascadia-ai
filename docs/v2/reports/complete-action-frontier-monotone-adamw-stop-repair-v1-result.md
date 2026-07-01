# Complete-Action Frontier Monotone AdamW Stop-Rule Repair V1 Result

Classification: `free_stage_passed`.

ADR 0108 reran only the five saturated ADR 0107 groups. The model, objective, optimizer, rates, moments, and strength gates remained unchanged. The other 19 groups and their replay evidence were reused byte-for-byte. Sealed test, gameplay, teacher, cloud, and external compute remained closed.

## Recombined Stage 1

| Checkpoint | Recall | Exact sets | Mean objective |
|---|---:|---:|---:|
| 120 updates | 96.24% | 70.83% | 2.920037 |
| terminal | 96.59% | 79.17% | 2.913515 |

All five repair groups met the frozen numerical-convergence rule.

| Group | Origin | Replay | Accepted | Recall | Exact | Smallest attempted rate |
|---:|---|---|---:|---:|---:|---:|
| 0 | john1 | john3 | 916 | 100.00% | yes | `5.899e-12` |
| 2 | john2 | john1 | 1105 | 100.00% | yes | `2.949e-12` |
| 8 | john3 | john4 | 767 | 97.67% | no | `2.949e-12` |
| 14 | john4 | john2 | 871 | 100.00% | yes | `5.899e-12` |
| 23 | john1 | john2 | 834 | 100.00% | yes | `7.373e-13` |

Every convergence event evaluated 16 finite proposals, retained finite parameters, moments, direction, and loss, and observed zero candidate improvement at float32 resolution.

## Frozen Gates

| Gate | Result |
|---|---|
| `all_19_frozen_replays_identical` | pass |
| `all_five_repair_replays_identical` | pass |
| `free_strength_gate_passed` | pass |
| `frozen_19_pipeline_passed` | pass |
| `frozen_lineage_passed` | pass |
| `recombined_pipeline_passed` | pass |
| `repair_pipeline_passed` | pass |

## Cluster Throughput

- End-to-end five-origin plus five-confirmation wall time: 6.71 seconds.
- Scheduled MLX process time: 22.55 seconds.
- Mean active MLX processes: 3.36; peak: 4.
- Idle process-slot seconds while compatible work was queued: 0.00.
- Duplicate discovery fraction: 0.00%; the five origins were distinct groups and duplication was limited to required cross-host confirmation.
- Source identity: 114 files, `ab432f3768d89642b93e25019fd078db5dafa16fefe1636a84fb30f29dbd4903`, identical on john1, john2, john3, john4.

## Authorized Successor

ADR 0107 neural Stage 2 is now authorized with the unchanged calibrated monotone AdamW mechanism: exactly four independent origins and four cross-host replays. A full trainer, validation treatment, sealed test, and gameplay remain closed.
