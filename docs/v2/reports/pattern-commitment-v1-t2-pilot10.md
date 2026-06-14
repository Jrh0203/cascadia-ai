# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `pattern-commitment-v1-k8-h6-b8-m4-t2`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.925
- Treatment mean: 91.250
- Baseline P10 / P50 / P90: 88.0 / 92.0 / 95.1
- Treatment P10 / P50 / P90: 87.0 / 91.0 / 96.0
- Baseline seat SD / range: 2.930 / 86.0-100.0
- Treatment seat SD / range: 3.240 / 85.0-98.0
- Paired delta: **-0.675**
- 95% CI: [-1.925, +0.575]
- Paired SD / SE: 2.017 / 0.638
- Game wins / ties / losses: 5 / 0 / 5
- Baseline decision latency mean / P50 / P90 / P99 / max: 5.93 / 4.41 / 12.28 / 23.56 / 49.03 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 11.91 / 10.42 / 21.75 / 43.08 / 142.35 ms
- Baseline runtime: 4.753s (0.475s/game)
- Treatment runtime: 9.548s (0.955s/game)
- Combined wall time: 1.556s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.600 | 5.875 | 5.775 | 5.975 | 5.600 |
| Treatment | 5.000 | 5.750 | 6.000 | 5.850 | 5.400 |
| Treatment - baseline | -0.600 | -0.125 | +0.225 | -0.125 | -0.200 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.750 | 11.100 | 12.050 | 12.500 | 14.775 | 3.925 |
| Treatment | 8.850 | 11.800 | 12.775 | 11.375 | 14.850 | 3.600 |
| Treatment - baseline | +0.100 | +0.700 | +0.725 | -1.125 | +0.075 | -0.325 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `16c688b2950512f701d6f1469fb599666894fba4501257f9fa68e2b7d66fcb05`
- Executable digest: `c4707b550a0959a85e07a8253ac6f7d799d8d248d7536dbe6131fcbbe54d8cbf`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "compare": {
    "baseline": "pattern-aware",
    "first_seed": 24100,
    "games": 10,
    "output": "docs/v2/reports/pattern-commitment-v1-t2-pilot10.json",
    "sequential": false,
    "treatment": "pattern-commitment"
  }
}
```
