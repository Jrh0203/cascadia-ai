# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `greedy-v1`
- Treatment: `determinized-lookahead-v2-k4-r4-d4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 86.880
- Treatment mean: 89.435
- Paired delta: **+2.555**
- 95% CI: [+1.915, +3.195]
- Paired SD / SE: 2.308 / 0.326
- Game wins / ties / losses: 44 / 1 / 5
- Runtime: 232.233s

## Mean Component Delta

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Treatment - baseline | +0.950 | -0.125 | -0.295 | -0.220 | +0.410 | +1.215 |
