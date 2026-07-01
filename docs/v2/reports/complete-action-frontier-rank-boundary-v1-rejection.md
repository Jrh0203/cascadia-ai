# Complete-Action Frontier Rank Boundary V1 Rejection

Status: **rejected on open validation**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-rank-boundary-v1`

## Verdict

Full rank-matched boundary supervision is rejected. Its direct score-space
ceiling and gradient coverage are perfect, but the shared neural
representation cannot improve the deployed target set under this objective.

| Metric | Selected | Gate |
|---|---:|---:|
| Train target recall | 29.36% | 60% |
| Train exact sets | 0.18% | 5% |
| Validation target recall | 26.21% | 50% |
| Validation exact sets | 0% | 1% |
| Validation winner recall | 76.67% | 75% |
| Validation confidence coverage | 90.42% | 90% |
| Validation regret | 0.061734 | <0.15 |

No trained epoch beat the warm start. Training loss fell from 1.1915 to
1.0667, while final validation target recall was 19.47%, winner recall 55.00%,
confidence coverage 77.92%, and regret 0.163564.

john3 verified complete rank-gradient coverage and a finite maximum-width
update. john4 recovered 100% exact sets on the 12 widest validation groups
inside ±12. All 93 MLX runtime files were byte-identical across four hosts;
237 tests and Ruff passed; swaps, sealed test, gameplay, and external compute
remained zero.

The next experiment is a frozen-embedding separability audit. It will decide
whether a small head can fit the boundary without changing the trunk,
separating optimizer coupling from missing representation before another
training pilot.
