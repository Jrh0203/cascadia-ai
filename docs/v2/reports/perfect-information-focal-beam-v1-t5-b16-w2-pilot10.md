# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-pattern-oracle-v1-k8-h6-b8-w2-m4`
- Treatment: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 92.900
- Treatment mean: 93.650
- Baseline P10 / P50 / P90: 88.0 / 93.0 / 97.0
- Treatment P10 / P50 / P90: 89.9 / 94.0 / 97.0
- Baseline seat SD / range: 3.311 / 85.0-99.0
- Treatment seat SD / range: 2.940 / 87.0-99.0
- Paired delta: **+0.750**
- 95% CI: [+0.400, +1.100]
- Paired SD / SE: 0.565 / 0.179
- Game wins / ties / losses: 9 / 1 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 132.77 / 126.19 / 224.42 / 488.28 / 1097.44 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1113.75 / 148.86 / 3252.17 / 14261.73 / 36269.76 ms
- Baseline runtime: 108.142s (10.814s/game)
- Treatment runtime: 893.059s (89.306s/game)
- Combined wall time: 1001.201s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.675 | 5.900 | 5.550 | 5.825 | 5.725 |
| Treatment | 5.450 | 6.100 | 5.325 | 6.075 | 5.975 |
| Treatment - baseline | -0.225 | +0.200 | -0.225 | +0.250 | +0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 18.650 | 6.675 | 12.300 | 9.125 | 13.900 | 3.575 |
| Treatment | 19.050 | 7.350 | 11.525 | 9.450 | 13.450 | 3.900 |
| Treatment - baseline | +0.400 | +0.675 | -0.775 | +0.325 | -0.450 | +0.325 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `9a2ac488021a979609cc7396c4d9418810b3cedf9b679c32aa0d9cd53952e6c5`
- Executable digest: `00e617be37ab4cff9b8b8d5efb0e0ed8f1327880a632dabe22409d9af0e4a96a`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-focal-beam-compare": {
    "beam_width": 16,
    "first_seed": 29600,
    "games": 10,
    "output": "docs/v2/reports/perfect-information-focal-beam-v1-t5-b16-w2-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "terminal_turns": 5,
    "wildlife_candidates": 2
  }
}
```
