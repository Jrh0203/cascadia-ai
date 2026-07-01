# Conditional Tile Optimizer Schedule V1 Result

Date: 2026-06-16

Experiment ID: `conditional-tile-optimizer-schedule-v1`

Classification: **`optimizer_schedule_tile_insufficient`**

## Tile Stage

| Metric | Fixed-rate source | Scheduled treatment | Delta | Gate |
|---|---:|---:|---:|---:|
| Train recall | 99.80% | 100.00% | +0.20% | >95% |
| Validation recall | 67.75% | 68.04% | +0.29% | >90% |
| Train exact queries | 96.45% | 100.00% | +3.55% | descriptive |
| Validation exact queries | 42.53% | 42.31% | -0.22% | descriptive |

Selected epoch: `191`. Origin elapsed:
`107.8 minutes`. Peak process RSS:
`3.00 GiB`.

The complete 200-epoch learning-rate trajectory matched
`hold20-cosine-to-3e-6-v1`: `True`.
Maximum train recall was `100.00%`.

## Action Proposal

| Metric | Validation | Gate |
|---|---:|---:|
| Tile-only oracle-stage target recall | 65.45% | >98% |
| Tile-only oracle-stage winner retention | 82.08% | >98% |
| Integrated proposal target recall | 64.94% | ADR 0115 |
| Integrated proposal winner retention | 82.08% | ADR 0115 |
| Integrated mean proposal count | 1061.8 | <=2,048 |

## Failed Gates

- `validation_tile_factor_recall_above_0_90`
- `mixed_validation_target_recall_above_0_98`
- `mixed_validation_winner_retention_above_0_98`
- `integrated_proposal_passed`
- `treatment_passed`

## Decision

The valid schedule treatment is insufficient. Close further exposure, sampling, and optimizer-schedule variants for this conditional pointwise tile ranker.
