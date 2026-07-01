# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `habitat-candidate-lookahead-v1-k8-h4-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 90.825
- Treatment mean: 90.575
- Baseline P10 / P50 / P90: 87.0 / 91.0 / 95.0
- Treatment P10 / P50 / P90: 87.0 / 90.5 / 95.0
- Baseline seat SD / range: 3.296 / 84.0-97.0
- Treatment seat SD / range: 3.079 / 85.0-97.0
- Paired delta: **-0.250**
- 95% CI: [-1.214, +0.714]
- Paired SD / SE: 1.555 / 0.492
- Game wins / ties / losses: 4 / 0 / 6
- Baseline runtime: 41.401s (4.140s/game)
- Treatment runtime: 57.568s (5.757s/game)
- Combined wall time: 98.969s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.450 | 6.025 | 5.800 | 5.700 | 5.875 |
| Treatment | 5.525 | 5.975 | 5.825 | 5.700 | 6.175 |
| Treatment - baseline | +0.075 | -0.050 | +0.025 | +0.000 | +0.300 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.475 | 12.800 | 12.825 | 12.525 | 14.700 | 3.650 |
| Treatment | 5.550 | 12.825 | 12.625 | 12.600 | 14.700 | 3.075 |
| Treatment - baseline | +0.075 | +0.025 | -0.200 | +0.075 | +0.000 | -0.575 |
