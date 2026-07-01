# Complete-Action Frontier Calibrated Neural Stage V1 Result

Classification: `calibrated_optimizer_pipeline_invalid`.

ADR 0109 applied the unchanged calibrated monotone AdamW mechanism to four frozen full-model local-fit groups. All four cross-host scientific replays were bit-identical. Sealed test, gameplay, new teacher compute, cloud, and external compute remained closed.

## Group Results

| Group | Origin | Replay | Accepted | Completion | Recall | Exact |
|---:|---|---|---:|---|---:|---:|
| 0 | john1 | john2 | 49 | numerically converged | 24.32% | 0.00% |
| 1 | john2 | john4 | 6 | numerically converged | 43.75% | 0.00% |
| 2 | john3 | john4 | 8 | failed: monotone AdamW could not accept an update | 24.39% | 0.00% |
| 3 | john4 | john3 | 1 | numerically converged | 40.62% | 0.00% |

Group 2 reproducibly failed the frozen completion rule after eight accepted updates. Its scores, moments, and accepted rates remained finite and it recorded zero nonfinite rejections, but it did not satisfy every preregistered numerical-convergence condition. The pipeline therefore fails before the strength classification is eligible.

The terminal descriptive aggregate was 32.39% recall and 0.00% exact sets. No group reached the 120-exposure checkpoint, so that checkpoint is correctly recorded as unobserved rather than fabricated.

## Frozen Gates

| Gate | Result |
|---|---|
| `all_four_replays_identical` | pass |
| `group_pipeline_passed` | fail |
| `neural_pipeline_passed` | fail |
| `strength_checkpoint_observed` | fail |
| `strength_gate_at_120_passed` | fail |
| `terminal_strength_gate_passed` | fail |

## Cluster Throughput

- End-to-end four-origin plus four-confirmation wall time: 31.31 seconds.
- Scheduled MLX process time: 51.58 seconds.
- Mean active MLX processes: 1.65; peak: 4.
- Idle process-slot seconds while compatible work was queued: 0.00.
- Duplicate discovery fraction: 0.00%; every origin tested a different group and all duplication was required confirmation.
- Source identity: 115 files, `ba8ec6aedcf24e5c00e717f69d57d9dabebc9e6aea1709343288f8f1e0087de4`, identical on john1-john4.

## Decision

The bounded full-trainer pilot is not authorized. The representation classification is also not eligible because the pipeline failed first. No additional neural compute may proceed under ADR 0109; any successor must be separately preregistered from the retained finite failure evidence.
