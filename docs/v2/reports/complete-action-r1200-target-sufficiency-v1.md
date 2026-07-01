# Complete-Action R1200 Target Sufficiency V1

Status: **target insufficient for set valued proposer**

## Result

| Metric | Train | Validation |
|---|---:|---:|
| R1200/R4800 95% set intersection | 98.75% | 97.08% |
| R1200 winner inside R4800 95% set | 88.39% | 89.17% |
| Mean R1200 cohort size | 961.37 | 959.86 |

## Validation Top 64

| Ranker | Exact winner | 95% set coverage | Regret |
|---|---:|---:|---:|
| Selected MLX model | 73.33% | 86.25% | 0.090184 |
| R1200 cohort oracle | 95.42% | 97.08% | 0.020742 |

## Phase

| Phase | Exact winner | 95% set coverage | Regret |
|---|---:|---:|---:|
| Early | 92.86% | 96.43% | 0.035737 |
| Late | 98.61% | 100.00% | 0.000216 |
| Middle | 95.24% | 95.24% | 0.023340 |

## Current Model Top-64 Composition

- R1200-labeled: 100.00%.
- R4800-labeled: 7.88%.
- Screen-only: 0.00%.

## Interpretation

Do not train the proposed cohort ranker; revise teacher allocation or observable representation.

The train and validation audit opened no sealed-test group. Validation
scientific metrics and complete rankings were identical on john1, john2,
and john3.
