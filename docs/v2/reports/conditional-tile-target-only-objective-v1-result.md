# Conditional Tile Target-Only Objective V1 Result

Date: 2026-06-16

Experiment ID: `conditional-tile-target-only-objective-v1`

Classification: **`target_only_tile_objective_insufficient`**

## Tile Stage

| Metric | Train | Validation | Gate |
|---|---:|---:|---:|
| Factor recall | 77.21% | 70.59% | >95% / >90% |
| Exact queries | 50.01% | 42.58% | descriptive |

Selected epoch: `20`. Peak process RSS:
`3.00 GiB`.
Process swaps: `0`.

## Action Proposal

| Metric | Validation | Gate |
|---|---:|---:|
| Tile-only oracle-stage target recall | 72.34% | >98% |
| Tile-only oracle-stage winner retention | 89.58% | >98% |
| Integrated proposal target recall | 71.83% | ADR 0115 |
| Integrated proposal winner retention | 89.58% | ADR 0115 |
| Integrated mean proposal count | 1062.0 | <=2,048 |

## Change From ADR 0115

| Metric | ADR 0115 | Target-only | Delta |
|---|---:|---:|---:|
| Train tile recall | 72.60% | 77.21% | +4.61% |
| Validation tile recall | 66.57% | 70.59% | +4.02% |
| Integrated proposal recall | 72.48% | 71.83% | -0.65% |
| Integrated winner retention | 92.08% | 89.58% | -2.50% |

## Integrity

- Selected weights replayed bit-identically on another host.
- The ADR 0115 source pipeline remained valid.
- Sealed test, gameplay, new teacher compute, cloud, and external compute
  remained closed.

## Failed Gates

- `train_tile_factor_recall_above_0_95`
- `validation_tile_factor_recall_above_0_90`
- `mixed_validation_target_recall_above_0_98`
- `mixed_validation_winner_retention_above_0_98`
- `integrated_proposal_passed`
- `treatment_passed`

## Decision

Boundary-only BCE is insufficient. Close this exact objective and audit model capacity and query-conditioned representation before another tile training run.
