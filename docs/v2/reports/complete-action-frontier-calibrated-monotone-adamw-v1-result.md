# Complete-Action Frontier Calibrated Monotone AdamW V1 Result

Classification: `calibrated_optimizer_pipeline_invalid`.

ADR 0107 Stage 1 applied one analytically capped, same-batch backtracked AdamW mechanism to the frozen 24 free-residual groups. Neural Stage 2 did not launch because the Stage 1 pipeline gate did not pass. Sealed test, gameplay, new teacher compute, cloud, and external compute remained closed.

## Free-Residual Result

| Checkpoint | Recall | Exact sets | Mean objective |
|---|---:|---:|---:|
| 120 updates | 96.24% | 70.83% | 2.920037 |
| terminal | 96.59% | 79.17% | 2.913661 |

The terminal strength gate passed, but the pipeline gate failed. Five groups reached float32 numerical saturation before the frozen 1,200-update count and could not accept another strictly loss-nonincreasing proposal within 16 backtracks.

| Group | Accepted updates | Recall | Exact | Min rate | Backtracks |
|---:|---:|---:|---:|---:|---:|
| 0 | 916 | 100.00% | yes | `9.664e-08` | 168 |
| 2 | 1105 | 100.00% | yes | `4.832e-08` | 45 |
| 8 | 767 | 97.67% | no | `4.832e-08` | 166 |
| 14 | 871 | 100.00% | yes | `9.664e-08` | 68 |
| 23 | 834 | 100.00% | yes | `1.208e-08` | 362 |

## Frozen Gates

| Gate | Result |
|---|---|
| `all_24_replays_identical` | pass |
| `free_pipeline_passed` | fail |
| `free_strength_gate_passed` | pass |

All 24 origin/replay scientific payloads were bit-identical and all resource gates passed.

## Cluster Throughput

- Successful origin-plus-confirmation wall time: 28.50 seconds.
- Successful scheduled process time: 111.75 seconds.
- Mean active MLX processes: 3.92; peak: 4.
- Idle process-slot seconds while compatible work was queued: 158.07; this includes the deliberate halt while the report bug was fixed, tested, synchronized, and source identity was refrozen.
- Pre-artifact implementation failures: 4 tasks, 9.13 process-seconds, caused by a report-field typo and rerun after source refreeze.
- Duplicate discovery fraction among retained artifacts: 0.00%; all duplicate work was explicit cross-host confirmation.
- Source identity: 113 files, `6664b99f0c37c08f78f97d7dd6730f1a37f1f45ed634da6f57bee1187c9ca958`, identical on john1, john2, john3, john4.

| Host | Tasks | Scheduled seconds |
|---|---:|---:|
| john1 | 15 | 28.41 |
| john2 | 11 | 27.45 |
| john3 | 11 | 28.47 |
| john4 | 11 | 27.42 |

## Authorized Successor

Preregister a stop-rule repair for only the five saturated groups. Treat exhausted finite backtracking as numerical convergence rather than requiring meaningless extra updates, then recombine with the frozen 19 completed groups. Neural work remains unauthorized until that repaired Stage 1 pipeline passes.
