# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 90.670
- Treatment mean: 91.760
- Baseline P10 / P50 / P90: 86.0 / 91.0 / 94.1
- Treatment P10 / P50 / P90: 88.0 / 92.0 / 96.0
- Baseline seat SD / range: 3.054 / 82.0-98.0
- Treatment seat SD / range: 3.018 / 83.0-101.0
- Paired delta: **+1.090**
- 95% CI: [+0.558, +1.622]
- Paired SD / SE: 1.920 / 0.272
- Game wins / ties / losses: 36 / 2 / 12
- Baseline runtime: 146.149s (2.923s/game)
- Treatment runtime: 240.602s (4.812s/game)
- Combined wall time: 386.751s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.380 | 5.775 | 5.955 | 5.860 | 6.035 |
| Treatment | 5.595 | 6.145 | 5.905 | 5.950 | 6.135 |
| Treatment - baseline | +0.215 | +0.370 | -0.050 | +0.090 | +0.100 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.615 | 12.495 | 13.000 | 12.990 | 15.390 | 3.175 |
| Treatment | 6.545 | 12.040 | 12.455 | 12.510 | 15.180 | 3.300 |
| Treatment - baseline | +1.930 | -0.455 | -0.545 | -0.480 | -0.210 | +0.125 |
