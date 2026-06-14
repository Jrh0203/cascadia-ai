# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `habitat-candidate-lookahead-v1-k8-h8-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 90.150
- Treatment mean: 91.725
- Baseline P10 / P50 / P90: 86.9 / 90.5 / 93.1
- Treatment P10 / P50 / P90: 88.0 / 92.0 / 95.0
- Baseline seat SD / range: 2.617 / 84.0-95.0
- Treatment seat SD / range: 3.063 / 85.0-100.0
- Paired delta: **+1.575**
- 95% CI: [+0.516, +2.634]
- Paired SD / SE: 1.708 / 0.540
- Game wins / ties / losses: 8 / 0 / 2
- Baseline runtime: 38.401s (3.840s/game)
- Treatment runtime: 76.295s (7.630s/game)
- Combined wall time: 114.696s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.200 | 5.975 | 5.700 | 5.650 | 6.225 |
| Treatment | 5.725 | 5.850 | 6.150 | 6.025 | 5.925 |
| Treatment - baseline | +0.525 | -0.125 | +0.450 | +0.375 | -0.300 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.750 | 12.975 | 12.800 | 12.475 | 15.175 | 3.225 |
| Treatment | 6.850 | 12.675 | 12.775 | 11.300 | 15.000 | 3.450 |
| Treatment - baseline | +2.100 | -0.300 | -0.025 | -1.175 | -0.175 | +0.225 |
