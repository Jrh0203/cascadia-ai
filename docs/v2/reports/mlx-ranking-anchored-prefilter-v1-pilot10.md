# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `mlx-anchored-prefilter-lookahead-v1-k8-b8-a6-p8-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 90.550
- Treatment mean: 91.425
- Baseline P10 / P50 / P90: 87.0 / 89.5 / 95.0
- Treatment P10 / P50 / P90: 88.0 / 92.0 / 95.1
- Baseline seat SD / range: 3.358 / 85.0-96.0
- Treatment seat SD / range: 3.388 / 79.0-98.0
- Paired delta: **+0.875**
- 95% CI: [-0.393, +2.143]
- Paired SD / SE: 2.045 / 0.647
- Game wins / ties / losses: 6 / 1 / 3
- Baseline runtime: 35.551s (3.555s/game)
- Treatment runtime: 39.959s (3.996s/game)
- Combined wall time: 75.510s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.350 | 5.825 | 5.925 | 6.025 | 5.500 |
| Treatment | 5.975 | 5.525 | 5.375 | 6.225 | 5.925 |
| Treatment - baseline | +0.625 | -0.300 | -0.550 | +0.200 | +0.425 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.100 | 12.875 | 13.175 | 12.150 | 15.275 | 3.350 |
| Treatment | 8.150 | 12.350 | 12.100 | 11.300 | 14.750 | 3.750 |
| Treatment - baseline | +3.050 | -0.525 | -1.075 | -0.850 | -0.525 | +0.400 |
