# P1 Relational Selected-Prefix Pointer Pilot v1 Result

**Completed:** 2026-06-17
**Experiment:** `p1-relational-selected-prefix-pointer-pilot-v1`
**Classification:** `p1_pointer_tile_stage_insufficient`

## Verdict

The exact sparse pointer foundation passed, but the learned P1 proposal pilot
failed.

All draft, tile, and wildlife stages completed their train-only checkpoint
selection, complete open-validation scoring, and exact distinct-host replay.
The frozen exact-R2 parent remained byte-identical. The learned tile pointer
recovered only 48.51% of validation target factors, far below the frozen 90%
stage gate. Integrated learned top-64 recall was 9.82% and R4800 winner
retention was 48.75%.

No sealed test or gameplay was opened.

## What Survived

- The active board fits in at most 82 exact sparse tokens in validation,
  comfortably below the 121-object contract.
- No 441-cell tensor is required for exact draft, tile, wildlife, action-map,
  or D6 pointer semantics.
- Every stage scored every query and item exactly once with finite logits.
- All three selected checkpoints reproduced exactly on a different Mac.
- Complete-action reconstruction took 1.46 ms per validation decision on
  average after stage scores were available.

## Validation

| Metric | Observed | Frozen gate |
|---|---:|---:|
| Draft target-factor recall | 83.86% | diagnostic |
| Tile target-factor recall | 48.51% | at least 90% |
| Wildlife target-factor recall | 100.00% | diagnostic |
| Mean proposal count | 998.75 | at most 1,024 |
| Oracle-inside-proposal target recall | 50.56% | greater than 98% |
| Oracle-inside-proposal winner retention | 80.83% | greater than 98% |
| Learned top-64 target recall | 9.82% | greater than 98% |
| Learned top-64 winner retention | 48.75% | greater than 98% |
| Learned top-64 confidence coverage | 72.50% | at least 99% |
| Learned top-64 retained regret | 0.1945 | below 0.15 |

The representation can name every exact target, but the fixed objective and
capacity do not identify the conditional tile target well enough from the
public query state. P1 is not authorized for gameplay or P2 expansion.

## Artifacts

- terminal integration:
  `artifacts/experiments/p1-relational-selected-prefix-pointer-pilot-v1/integration.json`;
- foundation classification:
  `artifacts/experiments/p1-relational-hierarchical-pointer-foundation-v1/classification.json`;
- selected stage runs:
  `artifacts/experiments/p1-relational-selected-prefix-pointer-pilot-v1/runs`;
- exact distinct-host replays:
  `artifacts/experiments/p1-relational-selected-prefix-pointer-pilot-v1/replays`.
