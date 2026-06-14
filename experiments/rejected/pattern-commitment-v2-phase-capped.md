# Pattern Commitment V2 Phase-Capped

Status: rejected on 2026-06-10.

## Hypothesis

Capping the two-turn wildlife Bellman value by the acting seat's realizable
remaining turns would remove rejected v1's impossible late-game setup value
while preserving useful early-game Bear commitment.

## Controlled Change

V2 kept the exact v1 candidate frontier, public wildlife supply, four-draw
market expectation, exact score transitions, and seeded tie breaking. After
each candidate action, it changed the opportunity horizon to
`min(2, acting-seat turns remaining)`.

A regression test proves that with one future turn remaining, the complete v2
candidate ranking equals the promoted one-turn policy exactly.

## Result

The seed-24399 runtime smoke passed at 0.478 treatment seconds per game. The
registered pilot on seeds 24400-24409 produced:

| Metric | Pattern-aware | Phase-capped commitment | Delta |
|---|---:|---:|---:|
| Mean base score | 91.575 | 92.225 | +0.650 |
| Bear | 7.200 | 9.100 | +1.900 |
| Aggregate non-Bear wildlife | 51.450 | 50.500 | -0.950 |
| Total wildlife | 58.650 | 59.600 | +0.950 |
| Habitat | 29.175 | 28.525 | -0.650 |
| Nature Tokens | 3.750 | 4.100 | +0.350 |
| Seconds per game | 0.275 | 0.338 | +0.063 |

The paired 95% confidence interval was `[-0.167, 1.467]`, with an 8-0-2
record.

## Conclusion

The score and Bear gates passed, but aggregate non-Bear wildlife and habitat
both exceeded the allowed -0.5 regression. No 50-game confirmation was run.
The stronger corrected signal is evidence that cross-turn Bear commitment
matters, while the failed guardrails show that an optimistic species-only
opportunity maximum still causes allocation tradeoffs rather than robust total
value creation.

Artifacts:

- `docs/v2/reports/pattern-commitment-v2-phase-capped-runtime-smoke-1.json`
- `docs/v2/reports/pattern-commitment-v2-phase-capped-pilot10.json`
