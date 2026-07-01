# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-root-diverse-beam-v1-t5-b16-rootw2-futurew2-k8-h6-b8-m4`
- Treatment: `perfect-information-root-diverse-beam-v1-t5-b16-rootw4-futurew2-k8-h6-b8-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 94.550
- Treatment mean: 94.625
- Baseline P10 / P50 / P90: 92.0 / 94.0 / 98.0
- Treatment P10 / P50 / P90: 92.0 / 94.0 / 98.0
- Baseline seat SD / range: 2.801 / 87.0-101.0
- Treatment seat SD / range: 2.780 / 87.0-101.0
- Paired delta: **+0.075**
- 95% CI: [-0.030, +0.180]
- Paired SD / SE: 0.169 / 0.053
- Game wins / ties / losses: 2 / 8 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 877.61 / 124.75 / 3108.91 / 11822.53 / 15577.91 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 992.52 / 124.50 / 3524.81 / 12936.44 / 20757.83 ms
- Baseline runtime: 703.676s (70.368s/game)
- Treatment runtime: 795.574s (79.557s/game)
- Combined wall time: 1499.251s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.100 | 7.075 | 4.975 | 5.825 | 6.250 |
| Treatment | 5.075 | 7.175 | 4.975 | 5.875 | 6.125 |
| Treatment - baseline | -0.025 | +0.100 | +0.000 | +0.050 | -0.125 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 16.725 | 8.775 | 10.350 | 9.300 | 16.275 | 3.900 |
| Treatment | 16.725 | 8.550 | 10.500 | 9.550 | 16.100 | 3.975 |
| Treatment - baseline | +0.000 | -0.225 | +0.150 | +0.250 | -0.175 | +0.075 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `e308d76cc7fe1365f13787c9409a4e8c78c029dbf9f3c5df01f3bd593b5f9924`
- Executable digest: `a096e297808dd2bf304730ec3c1e20c43f8b3f48fbbef3e56c4c1e32fbdd4c33`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-root-diverse-beam-compare": {
    "baseline_root_wildlife_candidates": 2,
    "beam_width": 16,
    "first_seed": 30700,
    "future_wildlife_candidates": 2,
    "games": 10,
    "output": "docs/v2/reports/perfect-information-root-diverse-beam-v1-t5-b16-rootw4-futurew2-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "terminal_turns": 5,
    "treatment_root_wildlife_candidates": 4
  }
}
```
