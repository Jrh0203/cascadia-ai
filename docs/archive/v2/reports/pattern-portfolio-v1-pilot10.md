# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `pattern-portfolio-v1-k8-h6-b8-m4-t2-conditioned-premium`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 92.550
- Treatment mean: 92.575
- Baseline P10 / P50 / P90: 89.0 / 93.0 / 97.0
- Treatment P10 / P50 / P90: 89.0 / 93.0 / 96.1
- Baseline seat SD / range: 3.012 / 85.0-98.0
- Treatment seat SD / range: 3.194 / 83.0-98.0
- Paired delta: **+0.025**
- 95% CI: [-1.291, +1.341]
- Paired SD / SE: 2.123 / 0.671
- Game wins / ties / losses: 4 / 0 / 6
- Baseline decision latency mean / P50 / P90 / P99 / max: 3.01 / 2.54 / 6.19 / 9.43 / 12.96 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 35.82 / 37.25 / 56.56 / 68.23 / 212.21 ms
- Baseline runtime: 2.417s (0.242s/game)
- Treatment runtime: 28.663s (2.866s/game)
- Combined wall time: 31.080s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.675 | 5.500 | 6.025 | 5.825 | 5.875 |
| Treatment | 5.350 | 6.000 | 6.075 | 5.625 | 5.875 |
| Treatment - baseline | -0.325 | +0.500 | +0.050 | -0.200 | +0.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.975 | 11.300 | 12.475 | 12.500 | 14.400 | 4.000 |
| Treatment | 8.400 | 11.700 | 12.275 | 12.175 | 15.025 | 4.075 |
| Treatment - baseline | -0.575 | +0.400 | -0.200 | -0.325 | +0.625 | +0.075 |

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
    "first_seed": 26300,
    "games": 10,
    "output": "docs/v2/reports/pattern-portfolio-v1-pilot10.json",
    "sequential": true,
    "treatment": "pattern-portfolio"
  }
}
```
