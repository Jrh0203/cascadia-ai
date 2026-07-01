# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `late-conservative-wildlife-focused-policy-improvement-v1-t5-r8-k8-h6-b8-fox2-m4-c90`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 89.250
- Treatment mean: 89.250
- Baseline P10 / P50 / P90: 85.9 / 89.0 / 92.8
- Treatment P10 / P50 / P90: 85.9 / 89.0 / 92.8
- Baseline seat SD / range: 3.775 / 85.0-94.0
- Treatment seat SD / range: 3.775 / 85.0-94.0
- Paired delta: **+0.000**
- 95% CI: [+0.000, +0.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 1 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 142.18 / 0.93 / 595.10 / 1376.99 / 1380.24 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 92.77 / 1.75 / 250.46 / 1322.33 / 1478.62 ms
- Baseline runtime: 11.375s (11.375s/game)
- Treatment runtime: 7.422s (7.422s/game)
- Combined wall time: 18.798s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.000 | 6.000 | 7.000 | 5.000 | 5.500 |
| Treatment | 6.000 | 6.000 | 7.000 | 5.000 | 5.500 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 6.750 | 10.500 | 12.000 | 13.500 | 13.750 | 3.250 |
| Treatment | 6.750 | 10.500 | 12.000 | 13.500 | 13.750 | 3.250 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

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
    "first_seed": 29299,
    "games": 1,
    "output": "docs/v2/reports/late-conservative-fox-frontier-v1-t5-r8-f2-c90-runtime-smoke-1.json",
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
