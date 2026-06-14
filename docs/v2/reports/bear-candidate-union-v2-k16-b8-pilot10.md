# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k16-r4-d4`
- Treatment: `bear-candidate-lookahead-v1-k16-b8-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.675
- Treatment mean: 91.525
- Baseline P10 / P50 / P90: 88.0 / 91.0 / 96.0
- Treatment P10 / P50 / P90: 87.7 / 91.5 / 96.0
- Baseline seat SD / range: 3.108 / 85.0-98.0
- Treatment seat SD / range: 3.968 / 80.0-100.0
- Paired delta: **-0.150**
- 95% CI: [-1.676, +1.376]
- Paired SD / SE: 2.461 / 0.778
- Game wins / ties / losses: 5 / 0 / 5
- Baseline runtime: 60.675s (6.067s/game)
- Treatment runtime: 68.952s (6.895s/game)
- Combined wall time: 129.626s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.725 | 5.875 | 5.850 | 6.200 | 6.000 |
| Treatment | 5.600 | 5.550 | 5.875 | 6.300 | 5.925 |
| Treatment - baseline | -0.125 | -0.325 | +0.025 | +0.100 | -0.075 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.325 | 11.775 | 12.800 | 13.000 | 15.550 | 3.575 |
| Treatment | 7.900 | 11.650 | 12.000 | 12.675 | 14.800 | 3.250 |
| Treatment - baseline | +2.575 | -0.125 | -0.800 | -0.325 | -0.750 | -0.325 |
