# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `perfect-information-pattern-oracle-v1-k8-h6-b8-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.750
- Treatment mean: 94.250
- Baseline P10 / P50 / P90: 91.3 / 92.5 / 94.4
- Treatment P10 / P50 / P90: 91.5 / 95.5 / 96.0
- Baseline seat SD / range: 1.708 / 91.0-95.0
- Treatment seat SD / range: 2.872 / 90.0-96.0
- Paired delta: **+1.500**
- 95% CI: [+1.500, +1.500]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 0.99 / 0.86 / 1.72 / 2.03 / 2.15 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 189.57 / 200.93 / 315.48 / 380.82 / 412.95 ms
- Baseline runtime: 0.079s (0.079s/game)
- Treatment runtime: 15.536s (15.536s/game)
- Combined wall time: 15.616s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 6.000 | 5.500 | 6.000 | 5.750 |
| Treatment | 7.250 | 5.500 | 6.000 | 6.250 | 5.500 |
| Treatment - baseline | +1.750 | -0.500 | +0.500 | +0.250 | -0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 10.500 | 12.750 | 11.500 | 9.500 | 15.500 | 4.250 |
| Treatment | 21.000 | 10.750 | 5.500 | 9.000 | 13.000 | 4.500 |
| Treatment - baseline | +10.500 | -2.000 | -6.000 | -0.500 | -2.500 | +0.250 |

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
    "first_seed": 28899,
    "games": 1,
    "output": "docs/v2/reports/perfect-information-frontier-bound-v1-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4
  }
}
```
