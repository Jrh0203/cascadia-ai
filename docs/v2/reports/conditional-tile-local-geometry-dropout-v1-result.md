# Conditional Tile Local-Geometry Dropout V1 Result

Date: 2026-06-16

Experiment ID: `conditional-tile-local-geometry-dropout-v1`

Classification: **`local_geometry_dropout_tile_insufficient`**

## Tile Stage

| Metric | ADR 0120 | Dropout treatment | Delta | Gate |
|---|---:|---:|---:|---:|
| Train recall | 100.00% | 98.89% | -1.11% | >95% |
| Validation recall | 68.04% | 67.16% | -0.88% | >90% |
| Train exact queries | 100.00% | 87.66% | -12.34% | descriptive |
| Validation exact queries | 42.31% | 42.06% | -0.25% | descriptive |

Selected epoch: `200`. Origin elapsed:
`109.0 minutes`. The complete
learning-rate trajectory matched `hold20-cosine-to-3e-6-v1`:
`True`. Exact dropout coverage matched
`epoch-hash-half-query-local-geometry-rotation-v1`: `True`.

## Action Proposal

| Metric | Validation | Gate |
|---|---:|---:|
| Tile-only oracle-stage target recall | 61.80% | >98% |
| Tile-only oracle-stage winner retention | 84.58% | >98% |
| Integrated proposal target recall | 61.27% | ADR 0115 |
| Integrated proposal winner retention | 84.58% | ADR 0115 |
| Integrated mean proposal count | 1062.1 | <=2,048 |

## Failed Gates

- `validation_tile_factor_recall_above_0_90`
- `mixed_validation_target_recall_above_0_98`
- `mixed_validation_winner_retention_above_0_98`
- `integrated_proposal_passed`
- `treatment_passed`

## Decision

The valid targeted regularizer is insufficient. Close this conditional pointwise tile representation and move upstream.
