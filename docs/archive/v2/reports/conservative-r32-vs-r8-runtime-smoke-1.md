# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `late-conservative-base-policy-improvement-v1-t5-r32-k8-h6-b8-m4-c90`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 88.750
- Treatment mean: 89.000
- Baseline P10 / P50 / P90: 87.3 / 88.5 / 90.4
- Treatment P10 / P50 / P90: 87.3 / 89.0 / 90.7
- Baseline seat SD / range: 1.708 / 87.0-91.0
- Treatment seat SD / range: 1.826 / 87.0-91.0
- Paired delta: **+0.250**
- 95% CI: [+0.250, +0.250]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 102.07 / 0.67 / 464.08 / 915.71 / 968.76 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 305.67 / 1.31 / 1158.35 / 2883.69 / 3490.99 ms
- Baseline runtime: 8.167s (8.167s/game)
- Treatment runtime: 24.454s (24.454s/game)
- Combined wall time: 32.621s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 5.250 | 6.000 | 5.250 | 7.500 |
| Treatment | 5.000 | 6.250 | 6.500 | 5.500 | 7.000 |
| Treatment - baseline | -0.500 | +1.000 | +0.500 | +0.250 | -0.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.500 | 9.500 | 11.250 | 12.750 | 13.250 | 3.000 |
| Treatment | 7.500 | 10.750 | 11.500 | 11.750 | 14.250 | 3.000 |
| Treatment - baseline | -2.000 | +1.250 | +0.250 | -1.000 | +1.000 | +0.000 |

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
    "first_seed": 28699,
    "games": 1,
    "output": "docs/v2/reports/conservative-r32-vs-r8-runtime-smoke-1.json",
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
