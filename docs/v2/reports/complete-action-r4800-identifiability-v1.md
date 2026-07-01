# Complete-Action R4800 Identifiability V1

Status: **representation or optimization material**

## Result

| Metric | Train | Validation |
|---|---:|---:|
| R4800 winner distinguishable at 95% | 32.14% | 31.67% |
| Mean 95% confidence-set size | 3.19 | 3.23 |
| R1200/R4800 argmax agreement | 63.75% | 66.67% |
| R1200 winner inside R4800 95% set | 90.18% | 90.00% |

## Validation Ranking

| Ranker | Exact winner recall | 95% set coverage | Retained regret |
|---|---:|---:|---:|
| Historical screen top 64 | 71.67% | 85.00% | 0.113024 |
| Selected MLX top 64 | 73.33% | 86.25% | 0.090184 |

Selected-model exact-winner misses retaining a statistically equivalent
action: 48.44%.

## Phase

| Phase | Distinguishable winner | Mean 95% set | Model set coverage | Regret |
|---|---:|---:|---:|---:|
| Early | 30.95% | 3.71 | 88.10% | 0.0934 |
| Late | 31.94% | 2.96 | 94.44% | 0.0247 |
| Middle | 32.14% | 2.98 | 77.38% | 0.1431 |

## Interpretation

Revise observable representation or optimization before continuation work.

The train and validation audit opened no sealed-test group. Validation
scientific metrics and complete action rankings were identical on john1,
john2, and john3.
