# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `mlx-prefilter-lookahead-v1-k8-b8-p8-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 90.275
- Treatment mean: 92.000
- Baseline P10 / P50 / P90: 86.0 / 91.0 / 93.1
- Treatment P10 / P50 / P90: 87.0 / 91.5 / 96.0
- Baseline seat SD / range: 3.242 / 80.0-96.0
- Treatment seat SD / range: 3.987 / 84.0-101.0
- Paired delta: **+1.725**
- 95% CI: [+0.528, +2.922]
- Paired SD / SE: 1.931 / 0.611
- Game wins / ties / losses: 7 / 2 / 1
- Baseline runtime: 48.848s (4.885s/game)
- Treatment runtime: 49.750s (4.975s/game)
- Combined wall time: 98.598s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.650 | 5.725 | 5.700 | 5.650 | 5.700 |
| Treatment | 5.600 | 5.875 | 5.925 | 5.800 | 5.975 |
| Treatment - baseline | -0.050 | +0.150 | +0.225 | +0.150 | +0.275 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.875 | 12.700 | 13.000 | 13.100 | 15.075 | 3.100 |
| Treatment | 7.575 | 11.975 | 13.100 | 12.600 | 13.650 | 3.925 |
| Treatment - baseline | +2.700 | -0.725 | +0.100 | -0.500 | -1.425 | +0.825 |
