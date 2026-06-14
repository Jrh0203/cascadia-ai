# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-wildlife-diverse-policy-improvement-v1-t5-r8-k8-h6-b8-w2-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.200
- Treatment mean: 91.750
- Baseline P10 / P50 / P90: 87.9 / 91.0 / 95.0
- Treatment P10 / P50 / P90: 88.0 / 92.0 / 95.0
- Baseline seat SD / range: 3.164 / 82.0-97.0
- Treatment seat SD / range: 2.880 / 86.0-99.0
- Paired delta: **+0.550**
- 95% CI: [-0.123, +1.223]
- Paired SD / SE: 1.085 / 0.343
- Game wins / ties / losses: 8 / 0 / 2
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.57 / 1.42 / 2.81 / 5.02 / 5.93 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 137.19 / 1.54 / 573.98 / 1446.09 / 3524.76 ms
- Baseline runtime: 1.268s (0.127s/game)
- Treatment runtime: 109.760s (10.976s/game)
- Combined wall time: 111.028s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.675 | 5.700 | 5.825 | 5.825 | 5.800 |
| Treatment | 5.725 | 6.000 | 5.900 | 5.800 | 5.900 |
| Treatment - baseline | +0.050 | +0.300 | +0.075 | -0.025 | +0.100 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.350 | 12.375 | 12.475 | 12.425 | 14.450 | 3.300 |
| Treatment | 8.975 | 11.900 | 11.825 | 12.225 | 14.250 | 3.250 |
| Treatment - baseline | +1.625 | -0.475 | -0.650 | -0.200 | -0.200 | -0.050 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `91d6a291b2bf8f350724f88c5f8ed3b1b304dcc300c0135a2d909ab5711aeede`
- Executable digest: `cf4f62c1ef6bc8d11d41b5216fb4a5cbeafc3c1ec4c54b278ad72aef6be20ca2`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "late-wildlife-diverse-policy-improvement-compare": {
    "determinizations": 8,
    "first_seed": 27300,
    "games": 10,
    "output": "docs/v2/reports/late-wildlife-diverse-policy-improvement-v1-t5-r8-w2-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5,
    "wildlife_candidates": 2
  }
}
```
