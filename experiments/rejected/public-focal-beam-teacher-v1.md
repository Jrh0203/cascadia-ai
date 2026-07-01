# Public Focal-Beam Teacher V1

Status: rejected after pilot on 2026-06-11.

## Hypothesis

Four shared public redeterminizations and a width-four focal beam would recover
enough of the exact beam's continuation gain to beat promoted strong and
qualify a multi-turn MLX teacher.

## Protocol

The treatment was pattern-aware before its final five personal turns. It then
evaluated K8+H6+B8+W2 candidates over four shared public redeterminizations,
used a width-four focal beam for every sampled continuation, and admitted only
challengers with a positive one-sided paired c90 lower bound.

## Result

| Metric | Strong | Public focal beam | Delta |
|---|---:|---:|---:|
| Mean base score | 92.925 | 92.850 | -0.075 |
| Habitat | 29.000 | 29.050 | +0.050 |
| Wildlife | 59.950 | 59.925 | -0.025 |
| Nature Tokens | 3.975 | 3.875 | -0.100 |

Paired 95% CI: `[-0.565,+0.415]`; record 5-2-3. Bear gained 0.475 and
aggregate non-Bear wildlife fell 0.500. Treatment runtime was 114.312 seconds
per game with 4,920 ms P90 decision latency.

## Conclusion

The treatment failed the +0.25 score and nonnegative total-wildlife gates.
No confirmation or MLX collection was permitted. This bounded public sampler
does not recover the exact beam's continuation gain.

Artifacts:

- `docs/archive/v2/reports/public-focal-beam-teacher-v1-t5-r4-b4-w2-c90-runtime-smoke-1.json`
- `docs/archive/v2/reports/public-focal-beam-teacher-v1-t5-r4-b4-w2-c90-pilot10.json`
