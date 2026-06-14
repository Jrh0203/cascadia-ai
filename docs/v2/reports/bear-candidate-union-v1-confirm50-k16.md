# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k16-r4-d4`
- Treatment: `bear-candidate-lookahead-v1-k8-b8-r4-d4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 91.610
- Treatment mean: 91.730
- Baseline P10 / P50 / P90: 87.0 / 92.0 / 95.0
- Treatment P10 / P50 / P90: 88.0 / 92.0 / 96.0
- Baseline seat SD / range: 2.953 / 83.0-98.0
- Treatment seat SD / range: 3.106 / 82.0-99.0
- Paired delta: **+0.120**
- 95% CI: [-0.346, +0.586]
- Paired SD / SE: 1.681 / 0.238
- Game wins / ties / losses: 24 / 3 / 23
- Baseline runtime: 280.112s (5.602s/game)
- Treatment runtime: 207.219s (4.144s/game)
- Combined wall time: 487.330s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.725 | 5.975 | 5.900 | 6.015 | 5.865 |
| Treatment | 5.645 | 5.975 | 5.855 | 5.765 | 5.875 |
| Treatment - baseline | -0.080 | +0.000 | -0.045 | -0.250 | +0.010 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.495 | 12.485 | 12.975 | 12.625 | 15.180 | 3.370 |
| Treatment | 8.055 | 11.730 | 12.390 | 12.270 | 14.555 | 3.615 |
| Treatment - baseline | +2.560 | -0.755 | -0.585 | -0.355 | -0.625 | +0.245 |
