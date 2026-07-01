# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-root-diverse-beam-v1-t5-b16-rootw2-futurew2-k8-h6-b8-m4`
- Treatment: `perfect-information-root-diverse-beam-v1-t5-b16-rootw4-futurew2-k8-h6-b8-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 93.500
- Treatment mean: 93.500
- Baseline P10 / P50 / P90: 88.7 / 95.5 / 96.7
- Treatment P10 / P50 / P90: 88.7 / 95.5 / 96.7
- Baseline seat SD / range: 5.066 / 86.0-97.0
- Treatment seat SD / range: 5.066 / 86.0-97.0
- Paired delta: **+0.000**
- 95% CI: [+0.000, +0.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 1 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 1215.52 / 198.82 / 3191.00 / 15706.45 / 19562.71 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1512.92 / 198.59 / 2868.08 / 22278.05 / 22943.02 ms
- Baseline runtime: 97.483s (97.483s/game)
- Treatment runtime: 121.296s (121.296s/game)
- Combined wall time: 218.780s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 4.750 | 5.750 | 3.750 | 5.500 | 7.250 |
| Treatment | 4.750 | 5.750 | 3.750 | 5.500 | 7.250 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 21.000 | 7.750 | 14.750 | 10.500 | 8.000 | 4.500 |
| Treatment | 21.000 | 7.750 | 14.750 | 10.500 | 8.000 | 4.500 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

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
    "first_seed": 30699,
    "future_wildlife_candidates": 2,
    "games": 1,
    "output": "docs/v2/reports/perfect-information-root-diverse-beam-v1-t5-b16-rootw4-futurew2-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "terminal_turns": 5,
    "treatment_root_wildlife_candidates": 4
  }
}
```
