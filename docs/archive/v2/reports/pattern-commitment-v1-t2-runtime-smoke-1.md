# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `pattern-commitment-v1-k8-h6-b8-m4-t2`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 91.500
- Treatment mean: 92.500
- Baseline P10 / P50 / P90: 88.6 / 91.5 / 94.4
- Treatment P10 / P50 / P90: 89.3 / 92.0 / 96.1
- Baseline seat SD / range: 3.109 / 88.0-95.0
- Treatment seat SD / range: 3.697 / 89.0-97.0
- Paired delta: **+1.000**
- 95% CI: [+1.000, +1.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 2.95 / 2.63 / 5.59 / 7.69 / 7.89 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 3.59 / 3.41 / 6.43 / 9.30 / 9.43 ms
- Baseline runtime: 0.237s (0.237s/game)
- Treatment runtime: 0.288s (0.288s/game)
- Combined wall time: 0.525s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 6.000 | 6.000 | 5.750 | 5.250 |
| Treatment | 5.500 | 6.250 | 4.750 | 6.000 | 6.000 |
| Treatment - baseline | +0.000 | +0.250 | -1.250 | +0.250 | +0.750 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.500 | 13.500 | 13.250 | 10.500 | 13.250 | 4.000 |
| Treatment | 9.500 | 11.250 | 10.750 | 11.750 | 16.250 | 4.500 |
| Treatment - baseline | +1.000 | -2.250 | -2.500 | +1.250 | +3.000 | +0.500 |

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
    "first_seed": 24099,
    "games": 1,
    "output": "docs/v2/reports/pattern-commitment-v1-t2-runtime-smoke-1.json",
    "sequential": false,
    "treatment": "pattern-commitment"
  }
}
```
