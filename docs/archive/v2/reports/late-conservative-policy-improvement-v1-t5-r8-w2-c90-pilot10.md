# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-conservative-policy-improvement-v1-t5-r8-k8-h6-b8-w2-m4-c90`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.775
- Treatment mean: 92.600
- Baseline P10 / P50 / P90: 88.8 / 91.0 / 97.1
- Treatment P10 / P50 / P90: 89.0 / 92.0 / 98.0
- Baseline seat SD / range: 3.690 / 82.0-100.0
- Treatment seat SD / range: 3.264 / 85.0-100.0
- Paired delta: **+0.825**
- 95% CI: [+0.452, +1.198]
- Paired SD / SE: 0.602 / 0.190
- Game wins / ties / losses: 9 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.23 / 1.12 / 2.23 / 2.97 / 3.39 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 111.88 / 1.15 / 483.79 / 1112.19 / 1518.63 ms
- Baseline runtime: 0.994s (0.099s/game)
- Treatment runtime: 89.509s (8.951s/game)
- Combined wall time: 90.503s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 5.900 | 5.725 | 5.500 | 5.700 |
| Treatment | 5.525 | 6.225 | 5.750 | 5.625 | 5.775 |
| Treatment - baseline | +0.025 | +0.325 | +0.025 | +0.125 | +0.075 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.800 | 12.525 | 12.250 | 12.350 | 14.500 | 4.025 |
| Treatment | 8.950 | 12.200 | 11.650 | 12.450 | 14.525 | 3.925 |
| Treatment - baseline | +1.150 | -0.325 | -0.600 | +0.100 | +0.025 | -0.100 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `8b19cd1be3561cf413e1058dfd4fd2fee0c341aad7916a3f7d20fc8b281ebbf5`
- Executable digest: `7a1d2e8ba3422ead75ce3e3d72d2b266e58f5a475378bd9a14cdd76c1b4caae2`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "late-conservative-policy-improvement-compare": {
    "first_seed": 27600,
    "games": 10,
    "output": "docs/v2/reports/late-conservative-policy-improvement-v1-t5-r8-w2-c90-pilot10.json",
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
