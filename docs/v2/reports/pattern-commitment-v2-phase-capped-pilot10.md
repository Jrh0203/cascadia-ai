# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `pattern-commitment-v2-k8-h6-b8-m4-t2-phase-capped`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.575
- Treatment mean: 92.225
- Baseline P10 / P50 / P90: 88.0 / 91.0 / 96.0
- Treatment P10 / P50 / P90: 89.0 / 92.0 / 96.0
- Baseline seat SD / range: 3.046 / 86.0-98.0
- Treatment seat SD / range: 2.806 / 86.0-100.0
- Paired delta: **+0.650**
- 95% CI: [-0.167, +1.467]
- Paired SD / SE: 1.319 / 0.417
- Game wins / ties / losses: 8 / 0 / 2
- Baseline decision latency mean / P50 / P90 / P99 / max: 3.43 / 2.46 / 7.65 / 13.51 / 25.93 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 4.22 / 3.54 / 8.27 / 11.79 / 17.30 ms
- Baseline runtime: 2.751s (0.275s/game)
- Treatment runtime: 3.379s (0.338s/game)
- Combined wall time: 0.660s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.625 | 5.725 | 5.725 | 5.800 | 6.300 |
| Treatment | 5.375 | 5.850 | 5.700 | 5.775 | 5.825 |
| Treatment - baseline | -0.250 | +0.125 | -0.025 | -0.025 | -0.475 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.200 | 12.050 | 12.750 | 11.650 | 15.000 | 3.750 |
| Treatment | 9.100 | 11.750 | 12.775 | 11.400 | 14.575 | 4.100 |
| Treatment - baseline | +1.900 | -0.300 | +0.025 | -0.250 | -0.425 | +0.350 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `e8d80e5d207a8fa1b153adb4528aecb4817646d56b8bbb80ce316a23e551d819`
- Executable digest: `a7e47b2f8373d6b4dde22c562821a54ccffedf7ab44ffb52c77167c45fe47e7e`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "compare": {
    "baseline": "pattern-aware",
    "first_seed": 24400,
    "games": 10,
    "output": "docs/v2/reports/pattern-commitment-v2-phase-capped-pilot10.json",
    "sequential": false,
    "treatment": "pattern-commitment"
  }
}
```
