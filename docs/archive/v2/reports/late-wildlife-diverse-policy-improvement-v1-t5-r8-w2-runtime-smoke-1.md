# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-wildlife-diverse-policy-improvement-v1-t5-r8-k8-h6-b8-w2-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.750
- Treatment mean: 94.000
- Baseline P10 / P50 / P90: 90.9 / 93.5 / 94.0
- Treatment P10 / P50 / P90: 91.2 / 95.0 / 96.0
- Baseline seat SD / range: 1.893 / 90.0-94.0
- Treatment seat SD / range: 2.828 / 90.0-96.0
- Paired delta: **+1.250**
- 95% CI: [+1.250, +1.250]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 0.93 / 0.89 / 1.39 / 1.75 / 1.89 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 136.25 / 0.92 / 521.70 / 1184.37 / 1918.63 ms
- Baseline runtime: 0.075s (0.075s/game)
- Treatment runtime: 10.901s (10.901s/game)
- Combined wall time: 10.976s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.750 | 6.500 | 5.500 | 6.250 | 5.000 |
| Treatment | 6.000 | 7.000 | 6.250 | 5.500 | 4.250 |
| Treatment - baseline | +0.250 | +0.500 | +0.750 | -0.750 | -0.750 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 6.750 | 11.750 | 12.750 | 12.000 | 15.750 | 4.750 |
| Treatment | 8.750 | 9.500 | 12.000 | 12.500 | 17.500 | 4.750 |
| Treatment - baseline | +2.000 | -2.250 | -0.750 | +0.500 | +1.750 | +0.000 |

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
    "first_seed": 27299,
    "games": 1,
    "output": "docs/v2/reports/late-wildlife-diverse-policy-improvement-v1-t5-r8-w2-runtime-smoke-1.json",
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
