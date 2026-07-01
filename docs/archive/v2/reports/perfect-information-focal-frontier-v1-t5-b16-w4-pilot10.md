# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Treatment: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w4-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 94.000
- Treatment mean: 93.075
- Baseline P10 / P50 / P90: 90.0 / 94.0 / 96.1
- Treatment P10 / P50 / P90: 88.9 / 93.5 / 96.0
- Baseline seat SD / range: 2.512 / 88.0-100.0
- Treatment seat SD / range: 2.921 / 88.0-98.0
- Paired delta: **-0.925**
- 95% CI: [-2.426, +0.576]
- Paired SD / SE: 2.421 / 0.766
- Game wins / ties / losses: 4 / 0 / 6
- Baseline decision latency mean / P50 / P90 / P99 / max: 803.72 / 120.96 / 2576.76 / 10034.21 / 25222.87 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1155.03 / 143.48 / 3402.95 / 14942.94 / 19982.29 ms
- Baseline runtime: 644.577s (64.458s/game)
- Treatment runtime: 925.521s (92.552s/game)
- Combined wall time: 1570.098s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.875 | 6.725 | 5.025 | 5.525 | 6.125 |
| Treatment | 5.325 | 6.150 | 5.650 | 5.125 | 6.625 |
| Treatment - baseline | -0.550 | -0.575 | +0.625 | -0.400 | +0.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 19.950 | 7.850 | 9.050 | 10.175 | 13.725 | 3.975 |
| Treatment | 18.700 | 7.350 | 8.775 | 10.225 | 15.150 | 4.000 |
| Treatment - baseline | -1.250 | -0.500 | -0.275 | +0.050 | +1.425 | +0.025 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `267989f90e3696a5ee69c207eada7df2baea8a9d4e0469c325c0aacdaf761150`
- Executable digest: `3ed2c68fcbdcc4b2d1fa6ef09b9260177efe0b478786191c35a868c546e29cb9`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-focal-frontier-compare": {
    "baseline_wildlife_candidates": 2,
    "beam_width": 16,
    "first_seed": 30500,
    "games": 10,
    "output": "docs/v2/reports/perfect-information-focal-frontier-v1-t5-b16-w4-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "terminal_turns": 5,
    "treatment_wildlife_candidates": 4
  }
}
```
