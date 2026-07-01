# Late Conservative Fox Frontier V1

Status: rejected after pilot on 2026-06-11.

## Hypothesis

The exact W2 diagnostic's +1.775 Fox gain would transfer to public play by
adding only two Fox-coverage actions to promoted strong's terminal frontier,
without reopening the all-species allocation failure.

## Controlled Change

The treatment preserved strong's final-five R8 c90 operator and complete
K8+H6+B8 frontier. It added at most two distinct Fox-drafting candidates.
Public redetermination, continuation, anchor, paired confidence rule, market
prelude, and tie handling were unchanged.

## Result

The runtime smoke passed at 7.422 treatment seconds. On seeds 29300-29309:

| Metric | Strong | Strong + Fox F2 | Delta |
|---|---:|---:|---:|
| Mean base score | 92.150 | 92.150 | 0.000 |
| Fox | 15.025 | 15.075 | +0.050 |
| Total wildlife | 59.300 | 59.275 | -0.025 |
| Non-Fox wildlife | 44.275 | 44.200 | -0.075 |
| Habitat | 28.500 | 28.500 | 0.000 |
| Nature Tokens | 4.350 | 4.375 | +0.025 |

All ten seed blocks tied. Treatment runtime was 5.525 seconds per game and
P90 decision latency was 344.233 ms.

## Conclusion

The treatment failed the +0.25 score and +0.25 Fox gates, so no confirmation
was permitted. The focused implementation is valid and fast, but the exact
oracle's candidate value did not transfer through the frozen R8 c90 public
estimator. Wider frontier work is closed until decision-local value
identification improves.

Artifacts:

- `docs/archive/v2/reports/late-conservative-fox-frontier-v1-t5-r8-f2-c90-runtime-smoke-1.json`
- `docs/archive/v2/reports/late-conservative-fox-frontier-v1-t5-r8-f2-c90-pilot10.json`
