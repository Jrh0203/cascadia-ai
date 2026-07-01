# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `mlx-ranking-v1-k8-b8`
- Games: 20 (80 seat scores per strategy)
- Baseline mean: 90.862
- Treatment mean: 88.638
- Baseline P10 / P50 / P90: 87.0 / 91.0 / 95.0
- Treatment P10 / P50 / P90: 84.0 / 89.0 / 93.0
- Baseline seat SD / range: 2.858 / 83.0-96.0
- Treatment seat SD / range: 3.702 / 80.0-96.0
- Paired delta: **-2.225**
- 95% CI: [-2.986, -1.464]
- Paired SD / SE: 1.736 / 0.388
- Game wins / ties / losses: 3 / 0 / 17
- Baseline runtime: 81.825s (4.091s/game)
- Treatment runtime: 10.493s (0.525s/game)
- Combined wall time: 92.318s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.925 | 5.862 | 5.862 | 5.638 | 5.725 |
| Treatment | 5.638 | 5.938 | 6.062 | 5.850 | 5.950 |
| Treatment - baseline | -0.287 | +0.075 | +0.200 | +0.212 | +0.225 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.763 | 12.700 | 13.062 | 12.875 | 14.775 | 3.675 |
| Treatment | 6.987 | 11.287 | 11.650 | 12.162 | 13.500 | 3.612 |
| Treatment - baseline | +2.225 | -1.412 | -1.412 | -0.713 | -1.275 | -0.062 |
