# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k16-r4-d4`
- Treatment: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 91.005
- Treatment mean: 91.520
- Baseline P10 / P50 / P90: 87.0 / 91.0 / 95.0
- Treatment P10 / P50 / P90: 87.0 / 92.0 / 95.1
- Baseline seat SD / range: 3.147 / 83.0-100.0
- Treatment seat SD / range: 3.256 / 82.0-100.0
- Paired delta: **+0.515**
- 95% CI: [-0.140, +1.170]
- Paired SD / SE: 2.364 / 0.334
- Game wins / ties / losses: 29 / 0 / 21
- Baseline runtime: 366.738s (7.335s/game)
- Treatment runtime: 311.786s (6.236s/game)
- Combined wall time: 678.524s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.730 | 5.955 | 5.840 | 6.015 | 5.905 |
| Treatment | 5.760 | 5.945 | 5.990 | 5.800 | 5.950 |
| Treatment - baseline | +0.030 | -0.010 | +0.150 | -0.215 | +0.045 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.925 | 12.460 | 12.765 | 12.800 | 15.305 | 3.305 |
| Treatment | 6.345 | 12.395 | 12.750 | 12.380 | 14.870 | 3.335 |
| Treatment - baseline | +1.420 | -0.065 | -0.015 | -0.420 | -0.435 | +0.030 |
