# Candidate Recall Diagnostic

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Trajectory policy: `determinized-lookahead-v2-k4-r4-d4`
- Expanded evaluator: `determinized-lookahead-v2-k8-r4-d4`
- Games / decisions: 5 / 400
- Selection coverage at K=4: 83.25%
- Value recall at K=4: 83.25%
- Strict value misses: 67
- Mean estimated regret: 0.076
- Mean regret when missed: 0.455
- Maximum estimated regret: 2.000
- Trajectory mean score: 89.200
- Runtime: 39.255s

## Phase Breakdown

| Phase | Decisions | Outside K | Strict misses | Value recall | Mean regret |
|---|---:|---:|---:|---:|---:|
| Early | 135 | 28 | 28 | 79.26% | 0.083 |
| Middle | 135 | 21 | 21 | 84.44% | 0.076 |
| Late | 130 | 18 | 18 | 86.15% | 0.069 |

Immediate-rank histogram: `[260, 38, 20, 15, 16, 13, 24, 14]`
