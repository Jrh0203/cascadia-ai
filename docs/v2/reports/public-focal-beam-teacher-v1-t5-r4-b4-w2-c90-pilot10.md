# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `late-conservative-focal-beam-v1-t5-r4-b4-k8-h6-b8-w2-m4-c90`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 92.925
- Treatment mean: 92.850
- Baseline P10 / P50 / P90: 89.0 / 93.0 / 97.1
- Treatment P10 / P50 / P90: 89.0 / 93.0 / 97.1
- Baseline seat SD / range: 3.238 / 87.0-100.0
- Treatment seat SD / range: 3.520 / 85.0-100.0
- Paired delta: **-0.075**
- 95% CI: [-0.565, +0.415]
- Paired SD / SE: 0.791 / 0.250
- Game wins / ties / losses: 5 / 2 / 3
- Baseline decision latency mean / P50 / P90 / P99 / max: 74.24 / 0.92 / 270.31 / 854.38 / 1615.08 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1428.89 / 0.99 / 4919.81 / 20286.63 / 39864.73 ms
- Baseline runtime: 59.401s (5.940s/game)
- Treatment runtime: 1143.121s (114.312s/game)
- Combined wall time: 1202.522s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.775 | 5.650 | 5.750 | 6.175 | 5.650 |
| Treatment | 5.675 | 5.825 | 5.775 | 5.975 | 5.800 |
| Treatment - baseline | -0.100 | +0.175 | +0.025 | -0.200 | +0.150 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.125 | 12.050 | 13.050 | 12.075 | 14.650 | 3.975 |
| Treatment | 8.600 | 11.600 | 12.900 | 12.075 | 14.750 | 3.875 |
| Treatment - baseline | +0.475 | -0.450 | -0.150 | +0.000 | +0.100 | -0.100 |

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
    "first_seed": 29800,
    "games": 10,
    "output": "docs/v2/reports/public-focal-beam-teacher-v1-t5-r4-b4-w2-c90-pilot10.json",
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
