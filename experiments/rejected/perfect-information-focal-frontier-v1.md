# Perfect-Information Focal Frontier V1

Status: rejected after pilot on 2026-06-11.

## Hypothesis

The exact final-five width-16 focal beam remained constrained by admitting
only two distinct actions per wildlife species. Widening that frontier to W4
would expose stronger multi-turn continuations.

## Protocol

Baseline and treatment used identical scalar width-16 exact focal beams,
final-five activation, hidden state, continuation randomness, opponents, and
terminal scoring. The only policy difference was W2 versus W4 wildlife
candidate coverage at every focal layer.

## Result

| Metric | W2 beam | W4 beam | Delta |
|---|---:|---:|---:|
| Mean base score | 94.000 | 93.075 | -0.925 |
| Habitat | 29.275 | 28.875 | -0.400 |
| Wildlife | 60.750 | 60.200 | -0.550 |
| Nature Tokens | 3.975 | 4.000 | +0.025 |

Paired 95% CI: `[-2.426,+0.576]`; record 4-0-6. Fox gained 1.425, but Bear
fell 1.250, Elk 0.500, Salmon 0.275, and total wildlife 0.550. Treatment
runtime was 92.552 seconds per four-seat block with 3,403 ms P90 latency.

The initial smoke missed runtime at 232.275 seconds. A score-identical general
engine optimization reduced it to 163.404 seconds, passing the gate and
preserving every score and category exactly.

## Conclusion

The promising +3.250 smoke was a false positive. W4 at every focal layer
regressed by 0.925 and remained below 97. Wider root recall and wider future
branching must be separated before action breadth is tested again.

Artifacts:

- `docs/archive/v2/reports/perfect-information-focal-frontier-v1-t5-b16-w4-runtime-smoke-1.json`
- `docs/archive/v2/reports/perfect-information-focal-frontier-v1-t5-b16-w4-runtime-smoke-2-optimized.json`
- `docs/archive/v2/reports/perfect-information-focal-frontier-v1-t5-b16-w4-pilot10.json`
