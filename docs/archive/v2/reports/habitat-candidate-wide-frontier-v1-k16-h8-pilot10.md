# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Treatment: `habitat-candidate-lookahead-v1-k16-h8-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 92.025
- Treatment mean: 91.700
- Baseline P10 / P50 / P90: 88.0 / 92.0 / 96.1
- Treatment P10 / P50 / P90: 89.0 / 91.0 / 95.0
- Baseline seat SD / range: 3.158 / 86.0-98.0
- Treatment seat SD / range: 2.839 / 85.0-97.0
- Paired delta: **-0.325**
- 95% CI: [-1.652, +1.002]
- Paired SD / SE: 2.141 / 0.677
- Game wins / ties / losses: 3 / 0 / 7
- Baseline decision latency mean / P50 / P90 / P99 / max: 78.71 / 65.49 / 159.28 / 239.93 / 293.50 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 129.30 / 107.16 / 256.18 / 396.43 / 518.54 ms
- Baseline runtime: 62.987s (6.299s/game)
- Treatment runtime: 103.453s (10.345s/game)
- Combined wall time: 166.439s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.400 | 6.225 | 6.025 | 5.650 | 6.200 |
| Treatment | 5.750 | 6.150 | 6.200 | 5.675 | 6.050 |
| Treatment - baseline | +0.350 | -0.075 | +0.175 | +0.025 | -0.150 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 6.525 | 12.100 | 12.700 | 13.125 | 14.700 | 3.375 |
| Treatment | 5.675 | 12.150 | 12.700 | 12.950 | 15.225 | 3.175 |
| Treatment - baseline | -0.850 | +0.050 | +0.000 | -0.175 | +0.525 | -0.200 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `8aad4c7d1509fc3236ea6e885b960a4dd1d73d892c60be3939bc97d0bcac7e02`
- Executable digest: `9d8163704541737c545e6dea7dabab1fb869befbf9d17e253d3e3a3f9248a5d3`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "habitat-candidate-ablate": {
    "baseline_candidates": 8,
    "baseline_determinizations": 4,
    "baseline_greedy_plies": 4,
    "baseline_habitat_candidates": 6,
    "first_seed": 23100,
    "games": 10,
    "output": "docs/v2/reports/habitat-candidate-wide-frontier-v1-k16-h8-pilot10.json",
    "treatment_candidates": 16,
    "treatment_determinizations": 4,
    "treatment_greedy_plies": 4,
    "treatment_habitat_candidates": 8
  }
}
```
