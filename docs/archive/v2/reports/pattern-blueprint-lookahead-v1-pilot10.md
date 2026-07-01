# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Treatment: `pattern-blueprint-lookahead-v1-k8-h6-r4-d4-pk8-ph6-pb8-pm4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.075
- Treatment mean: 90.525
- Baseline P10 / P50 / P90: 88.0 / 91.0 / 94.1
- Treatment P10 / P50 / P90: 86.9 / 90.0 / 95.1
- Baseline seat SD / range: 2.683 / 85.0-96.0
- Treatment seat SD / range: 3.427 / 83.0-98.0
- Paired delta: **-0.550**
- 95% CI: [-1.796, +0.696]
- Paired SD / SE: 2.010 / 0.636
- Game wins / ties / losses: 4 / 0 / 6
- Baseline decision latency mean / P50 / P90 / P99 / max: 64.78 / 54.71 / 127.28 / 227.45 / 334.04 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 97.53 / 83.82 / 173.99 / 419.18 / 564.88 ms
- Baseline runtime: 51.840s (5.184s/game)
- Treatment runtime: 78.033s (7.803s/game)
- Combined wall time: 129.873s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.800 | 5.775 | 5.750 | 5.975 | 5.975 |
| Treatment | 5.525 | 5.525 | 6.100 | 6.200 | 5.875 |
| Treatment - baseline | -0.275 | -0.250 | +0.350 | +0.225 | -0.100 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 6.775 | 12.475 | 12.300 | 11.900 | 15.175 | 3.175 |
| Treatment | 6.100 | 12.425 | 12.275 | 12.950 | 14.475 | 3.075 |
| Treatment - baseline | -0.675 | -0.050 | -0.025 | +1.050 | -0.700 | -0.100 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `b944cab99acb2502eab0e8f652880e041d86a4cc7c35487d62b8398453f806c9`
- Executable digest: `d55d93f9402fbb0202ea89de73961dd2304f1908c7000f2c81b8800512fab8c5`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "pattern-blueprint-compare": {
    "candidates": 8,
    "determinizations": 4,
    "first_seed": 23500,
    "games": 10,
    "habitat_candidates": 6,
    "output": "docs/v2/reports/pattern-blueprint-lookahead-v1-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "rollout_plies": 4
  }
}
```
