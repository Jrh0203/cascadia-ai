# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `bear-candidate-lookahead-v1-k8-b8-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 90.250
- Treatment mean: 92.250
- Baseline P10 / P50 / P90: 86.8 / 90.0 / 94.2
- Treatment P10 / P50 / P90: 86.8 / 92.0 / 97.2
- Baseline seat SD / range: 3.868 / 83.0-99.0
- Treatment seat SD / range: 4.211 / 85.0-101.0
- Paired delta: **+2.000**
- 95% CI: [+0.309, +3.691]
- Paired SD / SE: 2.728 / 0.863
- Game wins / ties / losses: 8 / 1 / 1
- Runtime: 82.716s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.550 | 5.825 | 5.975 | 5.600 | 5.925 |
| Treatment | 5.500 | 5.675 | 5.725 | 5.925 | 6.275 |
| Treatment - baseline | -0.050 | -0.150 | -0.250 | +0.325 | +0.350 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 3.450 | 13.150 | 13.425 | 12.650 | 15.500 | 3.200 |
| Treatment | 8.200 | 12.000 | 11.950 | 12.675 | 14.650 | 3.675 |
| Treatment - baseline | +4.750 | -1.150 | -1.475 | +0.025 | -0.850 | +0.475 |
