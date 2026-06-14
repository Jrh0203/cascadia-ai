# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `greedy-v1`
- Treatment: `pattern-aware-v1-k8-h6-b8-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 88.600
- Treatment mean: 92.175
- Baseline P10 / P50 / P90: 85.0 / 89.0 / 92.0
- Treatment P10 / P50 / P90: 88.9 / 92.5 / 96.0
- Baseline seat SD / range: 2.697 / 82.0-93.0
- Treatment seat SD / range: 2.863 / 86.0-97.0
- Paired delta: **+3.575**
- 95% CI: [+2.909, +4.241]
- Paired SD / SE: 1.074 / 0.340
- Game wins / ties / losses: 10 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 2.24 / 1.58 / 4.93 / 9.99 / 15.30 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 28.60 / 19.72 / 70.06 / 109.00 / 162.36 ms
- Baseline runtime: 1.804s (0.180s/game)
- Treatment runtime: 22.909s (2.291s/game)
- Combined wall time: 2.710s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 5.800 | 5.600 | 5.575 | 5.575 |
| Treatment | 5.900 | 5.925 | 5.800 | 5.575 | 5.775 |
| Treatment - baseline | +0.400 | +0.125 | +0.200 | +0.000 | +0.200 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.650 | 12.425 | 13.450 | 12.975 | 14.750 | 2.300 |
| Treatment | 8.775 | 11.600 | 12.375 | 12.500 | 14.375 | 3.575 |
| Treatment - baseline | +4.125 | -0.825 | -1.075 | -0.475 | -0.375 | +1.275 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `53cb93b345c8e994c7a459c064c9b563efe1852aea9fb9e769f5a6446bdedb6a`
- Executable digest: `7b4a10db48baa98b9da64b53110e85334f0cfe89096dee31884f7ca4a4306fc8`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "compare": {
    "baseline": "greedy",
    "first_seed": 23200,
    "games": 10,
    "output": "docs/v2/reports/pattern-aware-v1-pilot10.json",
    "sequential": false,
    "treatment": "pattern-aware"
  }
}
```
