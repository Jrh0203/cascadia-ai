# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-pattern-oracle-v1-k8-h6-b8-m4`
- Treatment: `perfect-information-pattern-oracle-v1-k8-h6-b8-w2-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 94.750
- Treatment mean: 94.250
- Baseline P10 / P50 / P90: 91.2 / 95.0 / 98.1
- Treatment P10 / P50 / P90: 93.3 / 94.0 / 95.4
- Baseline seat SD / range: 3.775 / 90.0-99.0
- Treatment seat SD / range: 1.258 / 93.0-96.0
- Paired delta: **-0.500**
- 95% CI: [-0.500, -0.500]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 151.33 / 143.48 / 257.27 / 318.87 / 383.22 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 178.10 / 165.44 / 295.61 / 411.04 / 469.36 ms
- Baseline runtime: 12.455s (12.455s/game)
- Treatment runtime: 14.558s (14.558s/game)
- Combined wall time: 27.013s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.000 | 5.750 | 5.750 | 7.000 | 5.750 |
| Treatment | 7.000 | 6.750 | 4.750 | 5.750 | 4.250 |
| Treatment - baseline | +2.000 | +1.000 | -1.000 | -1.250 | -1.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 17.000 | 6.250 | 13.000 | 7.250 | 17.000 | 5.000 |
| Treatment | 19.000 | 8.500 | 12.500 | 6.500 | 15.000 | 4.250 |
| Treatment - baseline | +2.000 | +2.250 | -0.500 | -0.750 | -2.000 | -0.750 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `2ef0b5b256f50309a52f9220ca7e6a5cff287b63f06fe939cdc81f4137c9b01b`
- Executable digest: `c4e01eb45123b8cb8542fbf2c0e2013f7d2eedfa48eb93274dbb77fde443c1a0`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-oracle-frontier-compare": {
    "first_seed": 29099,
    "games": 1,
    "output": "docs/v2/reports/perfect-information-diverse-frontier-v1-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "wildlife_candidates": 2
  }
}
```
