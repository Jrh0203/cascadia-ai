# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-pattern-oracle-v1-k8-h6-b8-m4`
- Treatment: `perfect-information-pattern-oracle-v1-k8-h6-b8-w2-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 92.625
- Treatment mean: 93.975
- Baseline P10 / P50 / P90: 88.0 / 93.0 / 97.0
- Treatment P10 / P50 / P90: 90.9 / 94.0 / 98.1
- Baseline seat SD / range: 3.192 / 86.0-98.0
- Treatment seat SD / range: 3.738 / 84.0-102.0
- Paired delta: **+1.350**
- 95% CI: [+0.704, +1.996]
- Paired SD / SE: 1.042 / 0.330
- Game wins / ties / losses: 9 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 161.71 / 149.80 / 300.52 / 435.84 / 723.33 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 189.08 / 181.22 / 337.53 / 497.48 / 629.24 ms
- Baseline runtime: 132.508s (13.251s/game)
- Treatment runtime: 154.489s (15.449s/game)
- Combined wall time: 286.997s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.750 | 6.425 | 5.300 | 5.525 | 6.325 |
| Treatment | 5.375 | 6.125 | 4.925 | 5.950 | 6.450 |
| Treatment - baseline | -0.375 | -0.300 | -0.375 | +0.425 | +0.125 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 17.025 | 8.950 | 12.300 | 8.325 | 13.450 | 3.250 |
| Treatment | 16.975 | 8.500 | 12.600 | 8.000 | 15.225 | 3.850 |
| Treatment - baseline | -0.050 | -0.450 | +0.300 | -0.325 | +1.775 | +0.600 |

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
    "first_seed": 29100,
    "games": 10,
    "output": "docs/v2/reports/perfect-information-diverse-frontier-v1-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "wildlife_candidates": 2
  }
}
```
