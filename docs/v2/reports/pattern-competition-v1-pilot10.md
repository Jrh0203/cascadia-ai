# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `pattern-competition-v1-k8-h6-b8-m4-t2-first-rotation`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.475
- Treatment mean: 92.350
- Baseline P10 / P50 / P90: 88.0 / 91.0 / 95.1
- Treatment P10 / P50 / P90: 87.0 / 93.0 / 96.0
- Baseline seat SD / range: 2.792 / 86.0-97.0
- Treatment seat SD / range: 3.262 / 85.0-97.0
- Paired delta: **+0.875**
- 95% CI: [-0.151, +1.901]
- Paired SD / SE: 1.655 / 0.523
- Game wins / ties / losses: 6 / 0 / 4
- Baseline decision latency mean / P50 / P90 / P99 / max: 6.17 / 4.42 / 13.49 / 29.04 / 47.58 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 128.91 / 130.54 / 228.59 / 341.63 / 416.35 ms
- Baseline runtime: 4.946s (0.495s/game)
- Treatment runtime: 103.163s (10.316s/game)
- Combined wall time: 11.392s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.375 | 5.450 | 5.775 | 5.900 | 5.800 |
| Treatment | 5.575 | 6.200 | 5.625 | 5.725 | 5.575 |
| Treatment - baseline | +0.200 | +0.750 | -0.150 | -0.175 | -0.225 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.500 | 11.975 | 12.500 | 12.500 | 15.125 | 3.575 |
| Treatment | 8.775 | 11.150 | 11.975 | 12.525 | 14.925 | 4.300 |
| Treatment - baseline | +1.275 | -0.825 | -0.525 | +0.025 | -0.200 | +0.725 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `c49be5e62bebdb68bf95d3c3b11615bb1f98c3ddfdcdeb1544103e07447c4a9e`
- Executable digest: `de4d32d0b279b5f7140e9ee9ab05d241d1ef523553919f4876117e1899bb8e62`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "compare": {
    "baseline": "pattern-aware",
    "first_seed": 26000,
    "games": 10,
    "output": "docs/v2/reports/pattern-competition-v1-pilot10.json",
    "sequential": false,
    "treatment": "pattern-competition"
  }
}
```
