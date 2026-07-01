# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `perfect-information-pattern-oracle-v1-k8-h6-b8-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.375
- Treatment mean: 93.150
- Baseline P10 / P50 / P90: 87.9 / 92.0 / 94.1
- Treatment P10 / P50 / P90: 89.0 / 93.5 / 98.0
- Baseline seat SD / range: 3.143 / 86.0-100.0
- Treatment seat SD / range: 3.231 / 86.0-100.0
- Paired delta: **+1.775**
- 95% CI: [+0.299, +3.251]
- Paired SD / SE: 2.382 / 0.753
- Game wins / ties / losses: 8 / 0 / 2
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.33 / 1.08 / 1.93 / 3.66 / 137.79 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 152.10 / 143.28 / 274.61 / 408.27 / 688.98 ms
- Baseline runtime: 1.071s (0.107s/game)
- Treatment runtime: 124.573s (12.457s/game)
- Combined wall time: 125.644s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.600 | 5.725 | 5.625 | 5.600 | 5.975 |
| Treatment | 5.425 | 6.450 | 5.525 | 5.100 | 6.400 |
| Treatment - baseline | -0.175 | +0.725 | -0.100 | -0.500 | +0.425 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.850 | 12.250 | 11.700 | 12.825 | 14.425 | 3.800 |
| Treatment | 19.175 | 7.275 | 11.650 | 8.775 | 13.850 | 3.525 |
| Treatment - baseline | +11.325 | -4.975 | -0.050 | -4.050 | -0.575 | -0.275 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `49830c8ef9af33db27b112968dd9f450d6233690ff49d3870823dc5e0dd87237`
- Executable digest: `5859375f93a96e4ff567fffaa29d98ba62daa48192a22b15b7bc159027c206ae`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-oracle-compare": {
    "first_seed": 28900,
    "games": 10,
    "output": "docs/v2/reports/perfect-information-frontier-bound-v1-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4
  }
}
```
