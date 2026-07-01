# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `pattern-portfolio-v1-k8-h6-b8-m4-t2-conditioned-premium`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 89.000
- Treatment mean: 92.000
- Baseline P10 / P50 / P90: 83.0 / 87.5 / 96.2
- Treatment P10 / P50 / P90: 88.2 / 91.5 / 96.2
- Baseline seat SD / range: 7.348 / 83.0-98.0
- Treatment seat SD / range: 4.546 / 87.0-98.0
- Paired delta: **+3.000**
- 95% CI: [+3.000, +3.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 2.18 / 1.73 / 4.63 / 5.92 / 6.13 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 35.05 / 35.99 / 54.11 / 62.79 / 64.85 ms
- Baseline runtime: 0.175s (0.175s/game)
- Treatment runtime: 2.805s (2.805s/game)
- Combined wall time: 2.980s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.000 | 6.000 | 6.000 | 6.000 | 5.500 |
| Treatment | 5.750 | 6.000 | 5.500 | 6.500 | 5.250 |
| Treatment - baseline | +0.750 | +0.000 | -0.500 | +0.500 | -0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.750 | 11.750 | 14.000 | 12.750 | 11.250 | 5.000 |
| Treatment | 11.250 | 9.250 | 13.250 | 11.000 | 14.000 | 4.250 |
| Treatment - baseline | +5.500 | -2.500 | -0.750 | -1.750 | +2.750 | -0.750 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `542d0774d7960a4e810d7b77983536973070c46e0e5f1848ab86b924263b5af4`
- Executable digest: `48829872c391643660af7ee01fddfbc65e113d93d0ae0f42a6217ba3b5da09ec`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "compare": {
    "baseline": "pattern-aware",
    "first_seed": 26299,
    "games": 1,
    "output": "docs/v2/reports/pattern-portfolio-v1-runtime-smoke-1.json",
    "sequential": true,
    "treatment": "pattern-portfolio"
  }
}
```
