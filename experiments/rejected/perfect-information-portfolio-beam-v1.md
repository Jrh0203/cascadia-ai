# Perfect-Information Portfolio Beam V1

Status: rejected after pilot on 2026-06-11.

## Hypothesis

The exact scalar beam discarded valuable multi-species continuations by
retaining every future layer with one pattern heuristic. Reserving width
across habitat, each wildlife score, Nature Tokens, and scalar value would
raise exact focal play.

## Protocol

The treatment changed only width-16 beam retention. Each future focal layer
nominated the top two evaluated children by scalar heuristic, habitat total,
Bear, Elk, Salmon, Hawk, Fox, and Nature Tokens in fixed order, collapsed
duplicate nominations, then filled unused capacity by scalar order. Hidden
state, W2 frontier, continuation randomness, opponents, final-five cutoff,
and exact terminal objective matched the scalar baseline.

## Result

| Metric | Scalar beam | Portfolio beam | Delta |
|---|---:|---:|---:|
| Mean base score | 94.025 | 94.075 | +0.050 |
| Habitat | 29.225 | 29.275 | +0.050 |
| Wildlife | 60.950 | 60.900 | -0.050 |
| Nature Tokens | 3.850 | 3.900 | +0.050 |

Paired 95% CI: `[-0.048,+0.148]`; record 1-9-0. Bear was unchanged, Elk
fell 0.050, Salmon gained 0.050, Hawk gained 0.075, and Fox fell 0.125.
Treatment runtime was 81.013 seconds per four-seat block with 3,596 ms P90
decision latency.

## Conclusion

The +0.050 gain is below the preregistered +0.10 mechanism threshold, and
nine of ten seed blocks tied exactly. Category-preserving retention is not
the missing continuation mechanism. The treatment remained below 97 and
5.925 points short of 100.

Artifacts:

- `docs/archive/v2/reports/perfect-information-portfolio-beam-v1-t5-b16-w2-runtime-smoke-1.json`
- `docs/archive/v2/reports/perfect-information-portfolio-beam-v1-t5-b16-w2-pilot10.json`
