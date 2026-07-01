# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Treatment: `pattern-blueprint-lookahead-v1-k8-h6-r4-d4-pk8-ph6-pb8-pm4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 90.500
- Treatment mean: 90.750
- Baseline P10 / P50 / P90: 87.6 / 90.0 / 93.8
- Treatment P10 / P50 / P90: 89.6 / 91.0 / 91.7
- Baseline seat SD / range: 3.416 / 87.0-95.0
- Treatment seat SD / range: 1.258 / 89.0-92.0
- Paired delta: **+0.250**
- 95% CI: [+0.250, +0.250]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 105.97 / 103.15 / 176.72 / 275.94 / 282.19 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 192.59 / 179.06 / 325.91 / 451.28 / 503.59 ms
- Baseline runtime: 8.479s (8.479s/game)
- Treatment runtime: 15.408s (15.408s/game)
- Combined wall time: 23.888s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.000 | 6.750 | 6.500 | 4.750 | 5.000 |
| Treatment | 6.000 | 5.250 | 6.500 | 5.750 | 5.250 |
| Treatment - baseline | +0.000 | -1.500 | +0.000 | +1.000 | +0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 12.250 | 10.750 | 12.750 | 17.250 | 3.000 |
| Treatment | 5.750 | 10.750 | 11.750 | 12.750 | 17.000 | 4.000 |
| Treatment - baseline | +0.250 | -1.500 | +1.000 | +0.000 | -0.250 | +1.000 |

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
    "first_seed": 23499,
    "games": 1,
    "habitat_candidates": 6,
    "output": "docs/v2/reports/pattern-blueprint-lookahead-v1-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "rollout_plies": 4
  }
}
```
