# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 90.125
- Treatment mean: 91.500
- Baseline P10 / P50 / P90: 86.0 / 90.0 / 94.1
- Treatment P10 / P50 / P90: 88.0 / 92.0 / 96.1
- Baseline seat SD / range: 3.244 / 85.0-98.0
- Treatment seat SD / range: 3.226 / 85.0-100.0
- Paired delta: **+1.375**
- 95% CI: [+0.116, +2.634]
- Paired SD / SE: 2.032 / 0.643
- Game wins / ties / losses: 8 / 0 / 2
- Baseline runtime: 45.873s (4.587s/game)
- Treatment runtime: 60.445s (6.045s/game)
- Combined wall time: 106.319s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.625 | 5.900 | 5.700 | 5.975 | 5.600 |
| Treatment | 5.900 | 5.900 | 6.150 | 6.200 | 5.850 |
| Treatment - baseline | +0.275 | +0.000 | +0.450 | +0.225 | +0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.200 | 12.625 | 13.025 | 13.200 | 15.325 | 2.950 |
| Treatment | 6.250 | 11.925 | 13.150 | 12.650 | 14.600 | 2.925 |
| Treatment - baseline | +2.050 | -0.700 | +0.125 | -0.550 | -0.725 | -0.025 |
