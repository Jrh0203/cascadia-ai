# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `late-conservative-focal-beam-v1-t5-r8-b16-k8-h6-b8-w2-m4-c90`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.000
- Treatment mean: 91.500
- Baseline P10 / P50 / P90: 90.3 / 91.5 / 94.1
- Treatment P10 / P50 / P90: 89.6 / 91.5 / 93.4
- Baseline seat SD / range: 2.160 / 90.0-95.0
- Treatment seat SD / range: 2.082 / 89.0-94.0
- Paired delta: **-0.500**
- 95% CI: [-0.500, -0.500]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 88.75 / 0.58 / 327.87 / 817.24 / 927.77 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 4614.57 / 1.38 / 7612.36 / 68022.17 / 72929.84 ms
- Baseline runtime: 7.101s (7.101s/game)
- Treatment runtime: 369.167s (369.167s/game)
- Combined wall time: 376.267s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.000 | 6.250 | 6.000 | 5.000 | 5.500 |
| Treatment | 5.000 | 6.000 | 5.500 | 4.750 | 5.250 |
| Treatment - baseline | +0.000 | -0.250 | -0.500 | -0.250 | -0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.500 | 11.750 | 11.750 | 12.250 | 13.750 | 5.250 |
| Treatment | 9.500 | 12.000 | 11.750 | 12.000 | 14.750 | 5.000 |
| Treatment - baseline | +0.000 | +0.250 | +0.000 | -0.250 | +1.000 | -0.250 |

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
    "first_seed": 31199,
    "games": 1,
    "output": "docs/v2/reports/public-focal-beam-oracle-v2-t5-r8-b16-w2-c90-runtime-smoke-1.json",
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
