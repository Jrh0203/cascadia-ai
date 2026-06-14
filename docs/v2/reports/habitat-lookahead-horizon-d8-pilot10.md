# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Treatment: `habitat-candidate-lookahead-v1-k8-h6-r4-d8`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.625
- Treatment mean: 91.575
- Baseline P10 / P50 / P90: 88.0 / 92.0 / 94.1
- Treatment P10 / P50 / P90: 88.0 / 92.0 / 94.1
- Baseline seat SD / range: 2.638 / 86.0-97.0
- Treatment seat SD / range: 2.978 / 82.0-98.0
- Paired delta: **-0.050**
- 95% CI: [-1.051, +0.951]
- Paired SD / SE: 1.615 / 0.511
- Game wins / ties / losses: 5 / 1 / 4
- Baseline runtime: 40.224s (4.022s/game)
- Treatment runtime: 67.723s (6.772s/game)
- Combined wall time: 107.948s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.700 | 5.800 | 5.675 | 6.000 | 6.200 |
| Treatment | 5.400 | 5.950 | 5.700 | 6.075 | 5.975 |
| Treatment - baseline | -0.300 | +0.150 | +0.025 | +0.075 | -0.225 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.075 | 12.725 | 12.100 | 12.175 | 14.675 | 3.500 |
| Treatment | 6.875 | 12.350 | 12.325 | 12.225 | 15.100 | 3.600 |
| Treatment - baseline | -0.200 | -0.375 | +0.225 | +0.050 | +0.425 | +0.100 |
