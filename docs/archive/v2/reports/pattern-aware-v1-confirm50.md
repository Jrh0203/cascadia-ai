# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `greedy-v1`
- Treatment: `pattern-aware-v1-k8-h6-b8-m4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 86.635
- Treatment mean: 91.525
- Baseline P10 / P50 / P90: 82.0 / 87.0 / 91.0
- Treatment P10 / P50 / P90: 87.0 / 92.0 / 95.0
- Baseline seat SD / range: 3.508 / 75.0-94.0
- Treatment seat SD / range: 3.075 / 84.0-100.0
- Paired delta: **+4.890**
- 95% CI: [+4.296, +5.484]
- Paired SD / SE: 2.144 / 0.303
- Game wins / ties / losses: 50 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 5.00 / 2.52 / 12.43 / 29.04 / 633.80 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 10.22 / 7.83 / 20.83 / 42.49 / 150.73 ms
- Baseline runtime: 20.132s (0.403s/game)
- Treatment runtime: 40.990s (0.820s/game)
- Combined wall time: 6.445s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.375 | 5.485 | 5.680 | 5.560 | 5.455 |
| Treatment | 5.660 | 5.605 | 5.920 | 5.860 | 5.805 |
| Treatment - baseline | +0.285 | +0.120 | +0.240 | +0.300 | +0.350 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 3.735 | 12.640 | 12.870 | 12.830 | 15.075 | 1.930 |
| Treatment | 7.610 | 11.875 | 12.575 | 12.175 | 14.645 | 3.795 |
| Treatment - baseline | +3.875 | -0.765 | -0.295 | -0.655 | -0.430 | +1.865 |

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
    "first_seed": 23300,
    "games": 50,
    "output": "docs/v2/reports/pattern-aware-v1-confirm50.json",
    "sequential": false,
    "treatment": "pattern-aware"
  }
}
```
