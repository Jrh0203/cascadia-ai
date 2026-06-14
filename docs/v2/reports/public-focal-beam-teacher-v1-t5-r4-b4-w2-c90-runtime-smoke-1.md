# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `late-conservative-focal-beam-v1-t5-r4-b4-k8-h6-b8-w2-m4-c90`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 91.500
- Treatment mean: 90.500
- Baseline P10 / P50 / P90: 88.3 / 91.5 / 94.7
- Treatment P10 / P50 / P90: 86.9 / 90.5 / 94.1
- Baseline seat SD / range: 3.512 / 88.0-95.0
- Treatment seat SD / range: 3.873 / 86.0-95.0
- Paired delta: **-1.000**
- 95% CI: [-1.000, -1.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 55.04 / 0.75 / 218.31 / 471.61 / 591.58 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1102.60 / 0.91 / 5897.88 / 10290.03 / 12495.65 ms
- Baseline runtime: 4.404s (4.404s/game)
- Treatment runtime: 88.208s (88.208s/game)
- Combined wall time: 92.612s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 4.500 | 6.500 | 6.250 | 6.750 | 6.000 |
| Treatment | 4.750 | 6.250 | 5.500 | 6.750 | 5.750 |
| Treatment - baseline | +0.250 | -0.250 | -0.750 | +0.000 | -0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 14.750 | 11.000 | 12.000 | 14.750 | 3.500 |
| Treatment | 5.500 | 14.750 | 11.000 | 12.750 | 14.500 | 3.000 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.750 | -0.250 | -0.500 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `2b3c0fd6dda8aac73eff734a8c022ffe3e1a16653334d85bc3189ed25d080670`
- Executable digest: `cf6d28f194dbdf7d733bb2e40ab61d29d1a1898a16a02d70dfc9e70246ff14ed`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "public-focal-beam-compare": {
    "beam_width": 4,
    "determinizations": 4,
    "first_seed": 29799,
    "games": 1,
    "output": "docs/v2/reports/public-focal-beam-teacher-v1-t5-r4-b4-w2-c90-runtime-smoke-1.json",
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
