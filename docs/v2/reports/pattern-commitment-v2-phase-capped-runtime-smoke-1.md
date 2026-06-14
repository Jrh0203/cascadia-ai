# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `pattern-commitment-v2-k8-h6-b8-m4-t2-phase-capped`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.500
- Treatment mean: 93.750
- Baseline P10 / P50 / P90: 89.9 / 92.5 / 95.1
- Treatment P10 / P50 / P90: 90.6 / 93.5 / 97.1
- Baseline seat SD / range: 2.887 / 89.0-96.0
- Treatment seat SD / range: 3.500 / 90.0-98.0
- Paired delta: **+1.250**
- 95% CI: [+1.250, +1.250]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 2.85 / 2.88 / 4.69 / 6.13 / 6.85 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 5.96 / 3.78 / 7.13 / 46.95 / 159.22 ms
- Baseline runtime: 0.229s (0.229s/game)
- Treatment runtime: 0.478s (0.478s/game)
- Combined wall time: 0.706s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.000 | 6.250 | 6.750 | 6.000 | 5.500 |
| Treatment | 5.000 | 7.750 | 5.750 | 5.750 | 6.000 |
| Treatment - baseline | +0.000 | +1.500 | -1.000 | -0.250 | +0.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.250 | 9.000 | 11.750 | 12.500 | 16.500 | 4.000 |
| Treatment | 7.500 | 10.250 | 15.000 | 12.750 | 14.750 | 3.250 |
| Treatment - baseline | -1.750 | +1.250 | +3.250 | +0.250 | -1.750 | -0.750 |

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
    "first_seed": 24399,
    "games": 1,
    "output": "docs/v2/reports/pattern-commitment-v2-phase-capped-runtime-smoke-1.json",
    "sequential": false,
    "treatment": "pattern-commitment"
  }
}
```
