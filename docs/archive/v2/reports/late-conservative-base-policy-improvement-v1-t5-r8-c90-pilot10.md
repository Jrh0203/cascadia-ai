# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 92.450
- Treatment mean: 92.950
- Baseline P10 / P50 / P90: 88.9 / 93.0 / 96.0
- Treatment P10 / P50 / P90: 87.9 / 93.0 / 96.1
- Baseline seat SD / range: 2.952 / 87.0-97.0
- Treatment seat SD / range: 3.396 / 86.0-102.0
- Paired delta: **+0.500**
- 95% CI: [-0.027, +1.027]
- Paired SD / SE: 0.850 / 0.269
- Game wins / ties / losses: 6 / 3 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.20 / 1.12 / 2.09 / 2.88 / 3.57 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 86.95 / 1.17 / 391.75 / 923.28 / 1510.30 ms
- Baseline runtime: 0.966s (0.097s/game)
- Treatment runtime: 69.564s (6.956s/game)
- Combined wall time: 70.530s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.250 | 5.875 | 6.150 | 5.675 | 5.925 |
| Treatment | 5.425 | 5.800 | 6.125 | 5.800 | 5.975 |
| Treatment - baseline | +0.175 | -0.075 | -0.025 | +0.125 | +0.050 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.075 | 12.600 | 11.875 | 12.350 | 14.500 | 4.175 |
| Treatment | 8.300 | 12.550 | 11.800 | 12.575 | 14.625 | 3.975 |
| Treatment - baseline | +0.225 | -0.050 | -0.075 | +0.225 | +0.125 | -0.200 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `b927a1b0154c199e102f249a2e54c24e4d286c45c3da3d6eff27116d87428155`
- Executable digest: `53d1cbe478951263af2133ce4a22788770d31ec601e22edf30715ddfa70d4237`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "late-conservative-base-policy-improvement-compare": {
    "first_seed": 27900,
    "games": 10,
    "output": "docs/v2/reports/late-conservative-base-policy-improvement-v1-t5-r8-c90-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5
  }
}
```
