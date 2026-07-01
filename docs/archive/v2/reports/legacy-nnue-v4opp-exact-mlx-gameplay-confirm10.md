# Exact MLX Gameplay Reproduction

Experiment:
`qualified-legacy-nnue-exact-mlx-gameplay-reproduction-v1-20260612`

## Result

Qualified every frozen gate on fresh paired gameplay seeds 32,600-32,609.

| Metric | Promoted strong | Exact MLX teacher | Delta |
|---|---:|---:|---:|
| Mean base score | 92.275 | 95.800 | +3.525 |
| Wildlife | 59.375 | 61.075 | +1.700 |
| Habitat | 28.875 | 30.575 | +1.700 |
| Nature Tokens | 4.025 | 4.150 | +0.125 |

Paired 95% confidence interval: `[+2.388,+4.662]`.

Record: 10 wins, zero ties, zero losses. The treatment's 95% interval was
`[94.843,96.757]`, P50 was 96, P90 was 100, and the observed range was
90-101.

## Integrity And Runtime

- 800/800 states translated;
- 800/800 selected actions legal;
- zero bridge or neural fallbacks;
- 39,886 neural batches and 63,217,274 neural rows;
- maximum neural batch 4,515 rows;
- 151.25 treatment seconds per game;
- 1.983-second median and 3.130-second P90 move latency;
- clean shutdown after all ten games.

The preceding smoke at seed 32,599 also passed, scoring 95.5 in 146.24
treatment seconds.

## Interpretation

The exact MLX search is a faithful, stable local reproduction of the
historical 95+ policy. It closes the Apple-neural migration risk and provides
a trustworthy research control, but it remains 4.2 points below the 100-point
target. Historical parameters are not promoted as the final V2 model.

Machine-readable report:
`legacy-nnue-v4opp-exact-mlx-gameplay-confirm10.json`

BLAKE3:
`79a6ec66fabccfe94e965ed2e3ae35e6050c637cf85d12f8627c4ff1b24fbec9`
