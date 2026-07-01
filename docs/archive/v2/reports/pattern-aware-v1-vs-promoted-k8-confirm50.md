# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `determinized-lookahead-v2-k8-r4-d4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 91.890
- Treatment mean: 90.775
- Baseline P10 / P50 / P90: 88.0 / 92.0 / 96.0
- Treatment P10 / P50 / P90: 87.0 / 91.0 / 95.0
- Baseline seat SD / range: 3.170 / 80.0-99.0
- Treatment seat SD / range: 3.005 / 84.0-98.0
- Paired delta: **-1.115**
- 95% CI: [-1.696, -0.534]
- Paired SD / SE: 2.098 / 0.297
- Game wins / ties / losses: 15 / 1 / 34
- Baseline decision latency mean / P50 / P90 / P99 / max: 3.64 / 3.18 / 6.86 / 11.53 / 48.49 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 52.82 / 45.42 / 105.13 / 161.26 / 300.67 ms
- Baseline runtime: 14.584s (0.292s/game)
- Treatment runtime: 211.340s (4.227s/game)
- Combined wall time: 225.924s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.530 | 5.915 | 5.790 | 5.885 | 5.830 |
| Treatment | 5.585 | 5.945 | 5.885 | 5.930 | 5.610 |
| Treatment - baseline | +0.055 | +0.030 | +0.095 | +0.045 | -0.220 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.910 | 11.660 | 12.415 | 12.035 | 14.945 | 3.975 |
| Treatment | 5.105 | 12.530 | 12.755 | 12.830 | 15.265 | 3.335 |
| Treatment - baseline | -2.805 | +0.870 | +0.340 | +0.795 | +0.320 | -0.640 |

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
  "lookahead-compare": {
    "baseline": "pattern-aware",
    "candidates": 8,
    "determinizations": 4,
    "first_seed": 23400,
    "games": 50,
    "greedy_plies": 4,
    "output": "docs/v2/reports/pattern-aware-v1-vs-promoted-k8-confirm50.json"
  }
}
```
