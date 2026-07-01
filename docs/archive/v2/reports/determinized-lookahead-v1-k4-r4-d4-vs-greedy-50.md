# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `greedy-v1`
- Treatment: `determinized-lookahead-v1-k4-r4-d4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 86.505
- Treatment mean: 89.815
- Paired delta: **+3.310**
- 95% CI: [+2.610, +4.010]
- Paired SD / SE: 2.526 / 0.357
- Game wins / ties / losses: 45 / 1 / 4
- Runtime: 281.431s

## Mean Component Delta

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Treatment - baseline | +1.065 | -0.175 | +0.415 | -0.175 | -0.125 | +1.375 |
