# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Treatment: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w4-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.250
- Treatment mean: 95.500
- Baseline P10 / P50 / P90: 89.5 / 93.5 / 94.0
- Treatment P10 / P50 / P90: 93.3 / 95.0 / 98.1
- Baseline seat SD / range: 2.872 / 88.0-94.0
- Treatment seat SD / range: 2.646 / 93.0-99.0
- Paired delta: **+3.250**
- 95% CI: [+3.250, +3.250]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 1552.26 / 235.74 / 3528.47 / 19145.68 / 22630.24 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 2898.35 / 413.70 / 8087.43 / 35307.19 / 42782.30 ms
- Baseline runtime: 124.489s (124.489s/game)
- Treatment runtime: 232.275s (232.275s/game)
- Combined wall time: 356.765s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 3.750 | 7.250 | 6.250 | 4.750 | 6.750 |
| Treatment | 4.250 | 8.000 | 4.750 | 5.750 | 6.250 |
| Treatment - baseline | +0.500 | +0.750 | -1.500 | +1.000 | -0.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 19.000 | 8.750 | 7.000 | 8.750 | 16.250 | 3.750 |
| Treatment | 16.250 | 9.000 | 11.750 | 8.750 | 16.500 | 4.250 |
| Treatment - baseline | -2.750 | +0.250 | +4.750 | +0.000 | +0.250 | +0.500 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `b628c1b0c220ad4431e00a7fb5d2cd57159a1de137435588ad70b1f10ea9e59b`
- Executable digest: `d9b6d6c1ebdac6c39881c6ade601cd5b25bfc163cd9393463d8e08d07e5eaeeb`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-focal-frontier-compare": {
    "baseline_wildlife_candidates": 2,
    "beam_width": 16,
    "first_seed": 30499,
    "games": 1,
    "output": "docs/v2/reports/perfect-information-focal-frontier-v1-t5-b16-w4-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "terminal_turns": 5,
    "treatment_wildlife_candidates": 4
  }
}
```
