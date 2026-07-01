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
- Baseline decision latency mean / P50 / P90 / P99 / max: 2.11 / 1.49 / 4.75 / 9.93 / 13.73 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 6.31 / 5.08 / 13.16 / 20.16 / 32.36 ms
- Baseline runtime: 1.701s (0.170s/game)
- Treatment runtime: 5.057s (0.506s/game)
- Combined wall time: 0.769s

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
- V2 source digest: `e5e3d563fec23dc7beeb9055626ae619491afa3638758a0847d59853738a89a2`
- Executable digest: `8284ec3c64ca07c50b9e38605f096f75cf88326482c4f8cfed9dc30e413d8f03`
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
    "output": "docs/v2/reports/pattern-aware-v1-unified-frontier-replication10.json",
    "sequential": false,
    "treatment": "pattern-aware"
  }
}
```
