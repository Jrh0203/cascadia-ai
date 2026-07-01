# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `pattern-competition-v1-k8-h6-b8-m4-t2-first-rotation`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 94.250
- Treatment mean: 94.000
- Baseline P10 / P50 / P90: 91.3 / 93.0 / 98.2
- Treatment P10 / P50 / P90: 91.0 / 93.5 / 97.4
- Baseline seat SD / range: 4.031 / 91.0-100.0
- Treatment seat SD / range: 3.559 / 91.0-98.0
- Paired delta: **-0.250**
- 95% CI: [-0.250, -0.250]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 2.20 / 2.23 / 4.23 / 5.63 / 6.65 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 65.07 / 66.00 / 96.70 / 104.80 / 105.87 ms
- Baseline runtime: 0.177s (0.177s/game)
- Treatment runtime: 5.206s (5.206s/game)
- Combined wall time: 5.383s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.250 | 5.000 | 6.750 | 6.500 | 6.250 |
| Treatment | 5.500 | 4.500 | 6.500 | 6.250 | 6.000 |
| Treatment - baseline | -0.750 | -0.500 | -0.250 | -0.250 | -0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 10.500 | 13.000 | 12.000 | 9.500 | 13.250 | 5.250 |
| Treatment | 12.500 | 11.250 | 14.250 | 10.250 | 12.250 | 4.750 |
| Treatment - baseline | +2.000 | -1.750 | +2.250 | +0.750 | -1.000 | -0.500 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `e338848ca99ac501ba5db02b8d78206a86da2f5bc02a3c3361e641fd045ecab1`
- Executable digest: `04f41256dc94290be71e9b387f9e8b6fca401337afdf2bba065e306e4de7070c`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "compare": {
    "baseline": "pattern-aware",
    "first_seed": 25999,
    "games": 1,
    "output": "docs/v2/reports/pattern-competition-v1-runtime-smoke-1.json",
    "sequential": false,
    "treatment": "pattern-competition"
  }
}
```
