# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Treatment: `perfect-information-portfolio-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 93.750
- Treatment mean: 93.750
- Baseline P10 / P50 / P90: 91.2 / 94.5 / 95.7
- Treatment P10 / P50 / P90: 91.2 / 94.5 / 95.7
- Baseline seat SD / range: 2.630 / 90.0-96.0
- Treatment seat SD / range: 2.630 / 90.0-96.0
- Paired delta: **+0.000**
- 95% CI: [+0.000, +0.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 1 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 1445.10 / 269.51 / 5201.34 / 14025.17 / 14804.46 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1356.68 / 225.66 / 3377.52 / 18141.67 / 25252.67 ms
- Baseline runtime: 115.961s (115.961s/game)
- Treatment runtime: 108.829s (108.829s/game)
- Combined wall time: 224.791s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.500 | 6.500 | 7.000 | 3.500 | 6.500 |
| Treatment | 6.500 | 6.500 | 7.000 | 3.500 | 6.500 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 15.250 | 5.000 | 10.000 | 11.250 | 18.750 | 3.500 |
| Treatment | 15.250 | 5.000 | 10.000 | 11.250 | 18.750 | 3.500 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

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
    "first_seed": 30299,
    "games": 1,
    "output": "docs/v2/reports/perfect-information-portfolio-beam-v1-t5-b16-w2-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "terminal_turns": 5,
    "wildlife_candidates": 2
  }
}
```
