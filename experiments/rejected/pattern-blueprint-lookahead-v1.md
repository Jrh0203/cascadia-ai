# Pattern-Aware Rollout Blueprint

Status: rejected on 2026-06-10.

## Hypothesis

H6's K8+H6 root frontier may be sound while its exact-greedy future policy
misprices durable wildlife setup. Replacing only those four future plies with
the confirmed pattern-aware policy should improve leaf estimates.

## Controlled Change

Both strategies used the identical K8+H6 root candidates, four
public-information determinizations, four future plies, and exact acting-seat
base score at the leaf. The treatment changed only future actions from exact
greedy to frozen `pattern-aware-v1-k8-h6-b8-m4`.

The full-config runtime smoke on seed 23499 passed at 15.408 treatment seconds
per game against the 60-second ceiling. That unlocked, but did not contribute
to, the registered pilot on seeds 23500-23509.

## Result

| Metric | H6 greedy rollout | Pattern rollout | Delta |
|---|---:|---:|---:|
| Mean base score | 91.075 | 90.525 | -0.550 |
| Habitat total | 29.275 | 29.225 | -0.050 |
| Wildlife total | 58.625 | 58.225 | -0.400 |
| Nature Tokens | 3.175 | 3.075 | -0.100 |
| Seconds per game | 5.184 | 7.803 | +2.619 |

The paired 95% confidence interval was `[-1.796, 0.696]`, with a 4-0-6 game
record. Bear fell 0.675 and Fox fell 0.700 while Hawk gained 1.050.

## Conclusion

The treatment missed the preregistered +0.25 score threshold, so no 50-game
confirmation was run. A policy that is stronger when acting directly is not
necessarily a better short-horizon rollout policy: its setup preferences can
remain unrealized at a four-ply exact-score leaf. Future policy iteration
should learn longer-term action values or distill search targets rather than
substituting this heuristic wholesale inside shallow rollouts.

Artifacts:

- `docs/v2/reports/pattern-blueprint-lookahead-v1-runtime-smoke-1.json`
- `docs/v2/reports/pattern-blueprint-lookahead-v1-pilot10.json`
