# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-conservative-policy-improvement-v1-t5-r8-k8-h6-b8-w2-m4-c90`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.000
- Treatment mean: 91.750
- Baseline P10 / P50 / P90: 89.3 / 91.5 / 95.1
- Treatment P10 / P50 / P90: 88.6 / 91.5 / 95.1
- Baseline seat SD / range: 3.162 / 89.0-96.0
- Treatment seat SD / range: 3.500 / 88.0-96.0
- Paired delta: **-0.250**
- 95% CI: [-0.250, -0.250]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.05 / 1.04 / 1.61 / 2.12 / 2.15 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 139.97 / 1.26 / 588.94 / 1333.96 / 1353.98 ms
- Baseline runtime: 0.085s (0.085s/game)
- Treatment runtime: 11.198s (11.198s/game)
- Combined wall time: 11.283s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.000 | 5.750 | 5.750 | 5.000 | 6.500 |
| Treatment | 6.250 | 6.250 | 5.500 | 4.500 | 7.000 |
| Treatment - baseline | +0.250 | +0.500 | -0.250 | -0.500 | +0.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.500 | 9.500 | 12.750 | 13.000 | 15.500 | 3.750 |
| Treatment | 8.500 | 10.250 | 12.750 | 12.000 | 15.000 | 3.750 |
| Treatment - baseline | +0.000 | +0.750 | +0.000 | -1.000 | -0.500 | +0.000 |

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
    "first_seed": 27599,
    "games": 1,
    "output": "docs/v2/reports/late-conservative-policy-improvement-v1-t5-r8-w2-c90-runtime-smoke-1.json",
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
