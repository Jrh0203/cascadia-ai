# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Treatment: `perfect-information-portfolio-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 94.025
- Treatment mean: 94.075
- Baseline P10 / P50 / P90: 90.8 / 94.0 / 98.0
- Treatment P10 / P50 / P90: 90.8 / 94.0 / 98.0
- Baseline seat SD / range: 2.769 / 88.0-99.0
- Treatment seat SD / range: 2.768 / 88.0-99.0
- Paired delta: **+0.050**
- 95% CI: [-0.048, +0.148]
- Paired SD / SE: 0.158 / 0.050
- Game wins / ties / losses: 1 / 9 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 997.40 / 139.47 / 3443.59 / 12653.95 / 18445.43 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1010.54 / 139.58 / 3595.92 / 13640.94 / 18540.83 ms
- Baseline runtime: 799.531s (79.953s/game)
- Treatment runtime: 810.134s (81.013s/game)
- Combined wall time: 1609.664s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.675 | 5.625 | 5.775 | 5.675 | 6.475 |
| Treatment | 5.700 | 5.625 | 5.775 | 5.675 | 6.500 |
| Treatment - baseline | +0.025 | +0.000 | +0.000 | +0.000 | +0.025 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 18.550 | 8.675 | 10.600 | 8.875 | 14.250 | 3.850 |
| Treatment | 18.550 | 8.625 | 10.650 | 8.950 | 14.125 | 3.900 |
| Treatment - baseline | +0.000 | -0.050 | +0.050 | +0.075 | -0.125 | +0.050 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `40b09c2f96b07b30632873daed0cbfcdb729141ff01e386ecf21144196ccf783`
- Executable digest: `542093a34cd00cf9d9e0f67c7b520201c8d81dfa469c5cea0e7293dfffabe5b2`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-portfolio-beam-compare": {
    "beam_width": 16,
    "first_seed": 30300,
    "games": 10,
    "output": "docs/v2/reports/perfect-information-portfolio-beam-v1-t5-b16-w2-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "terminal_turns": 5,
    "wildlife_candidates": 2
  }
}
```
