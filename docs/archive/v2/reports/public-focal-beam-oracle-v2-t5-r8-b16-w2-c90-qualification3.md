# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `late-conservative-focal-beam-v1-t5-r8-b16-k8-h6-b8-w2-m4-c90`
- Games: 3 (12 seat scores per strategy)
- Baseline mean: 92.500
- Treatment mean: 92.167
- Baseline P10 / P50 / P90: 90.0 / 93.0 / 96.0
- Treatment P10 / P50 / P90: 87.0 / 93.5 / 96.0
- Baseline seat SD / range: 3.317 / 86.0-97.0
- Treatment seat SD / range: 3.881 / 86.0-97.0
- Paired delta: **-0.333**
- 95% CI: [-0.987, +0.320]
- Paired SD / SE: 0.577 / 0.333
- Game wins / ties / losses: 0 / 2 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 41.70 / 0.55 / 176.58 / 408.85 / 444.13 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 2501.78 / 0.55 / 8291.95 / 34879.89 / 43701.83 ms
- Baseline runtime: 10.010s (3.337s/game)
- Treatment runtime: 600.428s (200.143s/game)
- Combined wall time: 610.438s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 6.333 | 5.250 | 6.000 | 6.583 |
| Treatment | 5.750 | 6.250 | 5.083 | 6.083 | 6.167 |
| Treatment - baseline | +0.250 | -0.083 | -0.167 | +0.083 | -0.417 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.000 | 12.083 | 12.167 | 12.167 | 13.917 | 3.500 |
| Treatment | 9.667 | 12.500 | 11.833 | 11.833 | 13.000 | 4.000 |
| Treatment - baseline | +0.667 | +0.417 | -0.333 | -0.333 | -0.917 | +0.500 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `4c034c1bcfb2391722b94c240b663d67cfe5d9c3cb4f73884965cf1b7e281bc1`
- Executable digest: `8c4c560aa6372fd26630fa0aaec12c5de81b01c39c971f0865d84bdd8e18b0b5`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "public-focal-beam-compare": {
    "beam_width": 16,
    "determinizations": 8,
    "first_seed": 31200,
    "games": 3,
    "output": "docs/v2/reports/public-focal-beam-oracle-v2-t5-r8-b16-w2-c90-qualification3.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5,
    "wildlife_candidates": 2
  }
}
```
