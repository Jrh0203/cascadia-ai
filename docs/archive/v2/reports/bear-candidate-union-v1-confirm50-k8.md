# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `bear-candidate-lookahead-v1-k8-b8-r4-d4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 90.350
- Treatment mean: 91.215
- Baseline P10 / P50 / P90: 86.0 / 91.0 / 94.0
- Treatment P10 / P50 / P90: 87.0 / 92.0 / 95.0
- Baseline seat SD / range: 3.019 / 80.0-99.0
- Treatment seat SD / range: 3.334 / 82.0-99.0
- Paired delta: **+0.865**
- 95% CI: [+0.320, +1.410]
- Paired SD / SE: 1.967 / 0.278
- Game wins / ties / losses: 31 / 5 / 14
- Baseline runtime: 204.795s (4.096s/game)
- Treatment runtime: 277.218s (5.544s/game)
- Combined wall time: 482.014s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 5.805 | 5.710 | 5.785 | 5.965 |
| Treatment | 5.470 | 5.860 | 5.730 | 5.915 | 5.830 |
| Treatment - baseline | -0.030 | +0.055 | +0.020 | +0.130 | -0.135 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.730 | 12.590 | 13.180 | 12.765 | 15.095 | 3.225 |
| Treatment | 8.290 | 11.915 | 12.455 | 12.195 | 14.170 | 3.385 |
| Treatment - baseline | +3.560 | -0.675 | -0.725 | -0.570 | -0.925 | +0.160 |
