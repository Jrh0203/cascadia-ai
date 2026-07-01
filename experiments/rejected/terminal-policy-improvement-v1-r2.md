# Terminal Policy Improvement V1 R2

Status: rejected on 2026-06-10.

## Hypothesis

Full-game continuation values would identify setup actions whose value survives
later markets and provide a stronger long-horizon teacher than H6's four-ply
exact-score labels.

## Controlled Change

The treatment retained the frozen pattern-aware K8+H6+B8 root frontier. Every
candidate used the same two public-information hidden-state
redeterminizations, followed by frozen pattern-aware play for all seats through
terminal score. The actual hidden stack was never scored.

## Result

The seed-24699 runtime smoke passed at 100.334 treatment seconds per game. The
registered seeds 24700-24702 produced:

| Metric | Pattern-aware | Terminal R2 | Delta |
|---|---:|---:|---:|
| Mean base score | 91.917 | 92.167 | +0.250 |
| Bear | 9.333 | 10.583 | +1.250 |
| Total wildlife | 58.583 | 59.750 | +1.167 |
| Habitat | 28.333 | 29.250 | +0.917 |
| Nature Tokens | 5.000 | 3.167 | -1.833 |
| Seconds per game | 0.193 | 50.545 | +50.352 |

The paired 95% confidence interval was `[-4.458, 4.958]`, with per-seed
deltas of -2.75, -1.50, and +5.00 and a 1-0-2 record.

## Conclusion

R2 failed the registered +1.0 teacher-qualification threshold, so terminal
dataset collection was not run. Unlike earlier Bear interventions, it improved
both wildlife and habitat, but two terminal samples were too noisy to produce
reliable policy improvement. Any higher-sample test requires a new experiment
ID and compute budget.

Artifacts:

- `docs/archive/v2/reports/terminal-policy-improvement-v1-r2-runtime-smoke-1.json`
- `docs/archive/v2/reports/terminal-policy-improvement-v1-r2-pilot3.json`
