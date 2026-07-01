# Conditional Tile Extended Exposure V1 Result

Date: 2026-06-16

Experiment ID: `conditional-tile-extended-exposure-v1`

Classification: **`extended_exposure_tile_insufficient`**

## Tile Stage

| Metric | 20 epochs | 200 epochs | Delta | Gate |
|---|---:|---:|---:|---:|
| Train recall | 77.21% | 99.80% | +22.60% | >95% |
| Validation recall | 70.59% | 67.75% | -2.84% | >90% |
| Train exact queries | 50.01% | 96.45% | +46.44% | descriptive |
| Validation exact queries | 42.58% | 42.53% | -0.05% | descriptive |

Selected epoch: `197`. Origin elapsed:
`108.5 minutes`. Peak process RSS:
`3.09 GiB`.

## Exposure Trajectory

- 80% train recall: `29`
- 85% train recall: `40`
- 90% train recall: `50`
- 95% train recall: `64`

Maximum train recall: `99.80%`.
Final epoch recall: `99.77%`.

## Action Proposal

| Metric | Validation | Gate |
|---|---:|---:|
| Tile-only oracle-stage target recall | 64.95% | >98% |
| Tile-only oracle-stage winner retention | 83.75% | >98% |
| Integrated proposal target recall | 64.42% | ADR 0115 |
| Integrated proposal winner retention | 83.75% | ADR 0115 |
| Integrated mean proposal count | 1061.8 | <=2,048 |

## Integrity

- The complete 200-epoch train-only trajectory is present and finite.
- Selected weights replayed bit-identically on john3.
- john4 owned the mixed ceiling and john1 owned integration.
- Sealed test, gameplay, new teacher compute, cloud, and external compute
  remained closed.

## Failed Gates

- `validation_tile_factor_recall_above_0_90`
- `mixed_validation_target_recall_above_0_98`
- `mixed_validation_winner_retention_above_0_98`
- `integrated_proposal_passed`
- `treatment_passed`

## Decision

Uniform full-data exposure is insufficient. Close pure epoch extension.
ADR 0119 also closed target-mass resampling, so the mechanical successor is
one frozen optimizer-schedule treatment.
