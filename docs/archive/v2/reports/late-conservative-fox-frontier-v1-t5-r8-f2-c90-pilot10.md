# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `late-conservative-wildlife-focused-policy-improvement-v1-t5-r8-k8-h6-b8-fox2-m4-c90`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 92.150
- Treatment mean: 92.150
- Baseline P10 / P50 / P90: 89.9 / 92.0 / 95.1
- Treatment P10 / P50 / P90: 89.9 / 92.0 / 95.1
- Baseline seat SD / range: 2.966 / 83.0-100.0
- Treatment seat SD / range: 3.085 / 83.0-100.0
- Paired delta: **+0.000**
- 95% CI: [+0.000, +0.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 10 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 68.59 / 1.12 / 322.64 / 688.01 / 1493.53 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 69.05 / 1.08 / 344.23 / 601.25 / 782.54 ms
- Baseline runtime: 54.875s (5.487s/game)
- Treatment runtime: 55.250s (5.525s/game)
- Combined wall time: 110.125s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.600 | 6.275 | 5.475 | 5.725 | 5.425 |
| Treatment | 5.625 | 6.275 | 5.475 | 5.700 | 5.425 |
| Treatment - baseline | +0.025 | +0.000 | +0.000 | -0.025 | +0.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.525 | 11.225 | 11.950 | 12.575 | 15.025 | 4.350 |
| Treatment | 8.425 | 11.225 | 12.050 | 12.500 | 15.075 | 4.375 |
| Treatment - baseline | -0.100 | +0.000 | +0.100 | -0.075 | +0.050 | +0.025 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `50bad9aba206e7ecbe435d2ac0011e339a4c8ed6d4163bd08bde660b74983fae`
- Executable digest: `87fec46634824c7b9013accf9ea09d26b6891ff306c361a73da08a24fd423d38`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "late-conservative-wildlife-focused-policy-improvement-compare": {
    "determinizations": 8,
    "first_seed": 29300,
    "games": 10,
    "output": "docs/v2/reports/late-conservative-fox-frontier-v1-t5-r8-f2-c90-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5,
    "wildlife": "fox",
    "wildlife_candidates": 2
  }
}
```
