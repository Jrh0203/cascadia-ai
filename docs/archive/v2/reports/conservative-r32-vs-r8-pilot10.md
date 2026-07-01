# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `late-conservative-base-policy-improvement-v1-t5-r32-k8-h6-b8-m4-c90`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.425
- Treatment mean: 91.300
- Baseline P10 / P50 / P90: 88.8 / 92.0 / 95.0
- Treatment P10 / P50 / P90: 87.9 / 91.0 / 95.0
- Baseline seat SD / range: 2.943 / 83.0-100.0
- Treatment seat SD / range: 3.244 / 83.0-100.0
- Paired delta: **-0.125**
- 95% CI: [-0.616, +0.366]
- Paired SD / SE: 0.793 / 0.251
- Game wins / ties / losses: 4 / 0 / 6
- Baseline decision latency mean / P50 / P90 / P99 / max: 91.60 / 1.58 / 403.81 / 891.88 / 1637.17 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 352.71 / 1.32 / 1552.50 / 3438.57 / 5745.49 ms
- Baseline runtime: 73.290s (7.329s/game)
- Treatment runtime: 282.175s (28.217s/game)
- Combined wall time: 355.466s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.400 | 5.875 | 5.850 | 6.025 | 5.800 |
| Treatment | 5.600 | 5.675 | 5.625 | 5.925 | 5.950 |
| Treatment - baseline | +0.200 | -0.200 | -0.225 | -0.100 | +0.150 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.025 | 11.575 | 12.250 | 11.925 | 15.150 | 3.550 |
| Treatment | 8.075 | 11.500 | 12.450 | 11.700 | 15.150 | 3.650 |
| Treatment - baseline | +0.050 | -0.075 | +0.200 | -0.225 | +0.000 | +0.100 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `19dc7ed9b7cbbf5b271ac59cb276e28213c765c967fff722d18445475b71ec29`
- Executable digest: `1e675ac66c4490d5f6ca39abd40ee7457528462779c7e97878142b7e0f6e3827`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "conservative-sample-count-compare": {
    "baseline_determinizations": 8,
    "first_seed": 28700,
    "games": 10,
    "output": "docs/v2/reports/conservative-r32-vs-r8-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5,
    "treatment_determinizations": 32
  }
}
```
