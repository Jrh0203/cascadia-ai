# Candidate Recall Diagnostic

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Trajectory policy: `determinized-lookahead-v2-k8-r4-d4`
- Expanded evaluator: `determinized-lookahead-v2-k16-r4-d4`
- Games / decisions: 3 / 240
- Selection coverage at K=8: 89.17%
- Value recall at K=8: 89.17%
- Strict value misses: 26
- Mean estimated regret: 0.055
- Mean regret when missed: 0.510
- Maximum estimated regret: 1.750
- Trajectory mean score: 91.917
- Runtime: 34.401s

## Phase Breakdown

| Phase | Decisions | Outside K | Strict misses | Value recall | Mean regret |
|---|---:|---:|---:|---:|---:|
| Early | 81 | 12 | 12 | 85.19% | 0.093 |
| Middle | 81 | 8 | 8 | 90.12% | 0.034 |
| Late | 78 | 6 | 6 | 92.31% | 0.038 |

Immediate-rank histogram: `[140, 23, 15, 12, 11, 3, 8, 2, 3, 3, 4, 1, 3, 3, 4, 5]`
