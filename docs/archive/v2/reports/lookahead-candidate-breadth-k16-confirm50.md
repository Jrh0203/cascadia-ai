# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `determinized-lookahead-v2-k8-r4-d4`
- Treatment: `determinized-lookahead-v2-k16-r4-d4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 90.810
- Treatment mean: 91.555
- Paired delta: **+0.745**
- 95% CI: [+0.187, +1.303]
- Paired SD / SE: 2.012 / 0.285
- Game wins / ties / losses: 33 / 1 / 16
- Runtime: 421.778s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.545 | 6.000 | 5.970 | 5.835 | 5.775 |
| Treatment | 5.700 | 5.995 | 6.010 | 5.765 | 5.790 |
| Treatment - baseline | +0.155 | -0.005 | +0.040 | -0.070 | +0.015 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.855 | 12.540 | 12.870 | 12.795 | 15.255 | 3.370 |
| Treatment | 5.000 | 12.630 | 12.885 | 13.005 | 15.295 | 3.480 |
| Treatment - baseline | +0.145 | +0.090 | +0.015 | +0.210 | +0.040 | +0.110 |
