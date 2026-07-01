# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `determinized-lookahead-v2-k16-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.000
- Treatment mean: 91.425
- Paired delta: **+0.425**
- 95% CI: [-1.079, +1.929]
- Paired SD / SE: 2.427 / 0.767
- Game wins / ties / losses: 5 / 0 / 5
- Runtime: 106.011s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.775 | 6.300 | 5.425 | 5.625 | 5.650 |
| Treatment | 5.700 | 6.450 | 5.800 | 5.900 | 5.750 |
| Treatment - baseline | -0.075 | +0.150 | +0.375 | +0.275 | +0.100 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.650 | 12.450 | 13.075 | 12.150 | 15.500 | 3.400 |
| Treatment | 5.700 | 12.850 | 12.700 | 11.600 | 15.775 | 3.200 |
| Treatment - baseline | +0.050 | +0.400 | -0.375 | -0.550 | +0.275 | -0.200 |
