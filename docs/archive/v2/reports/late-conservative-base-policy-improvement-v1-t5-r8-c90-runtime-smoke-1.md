# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 94.250
- Treatment mean: 94.500
- Baseline P10 / P50 / P90: 90.8 / 95.0 / 97.1
- Treatment P10 / P50 / P90: 91.5 / 95.5 / 96.7
- Baseline seat SD / range: 3.775 / 89.0-98.0
- Treatment seat SD / range: 3.109 / 90.0-97.0
- Paired delta: **+0.250**
- 95% CI: [+0.250, +0.250]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 0.88 / 0.87 / 1.33 / 1.55 / 1.59 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 139.04 / 0.91 / 543.69 / 1289.67 / 1407.16 ms
- Baseline runtime: 0.071s (0.071s/game)
- Treatment runtime: 11.124s (11.124s/game)
- Combined wall time: 11.195s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.000 | 6.500 | 5.000 | 6.000 | 5.750 |
| Treatment | 6.250 | 6.750 | 5.250 | 5.500 | 6.250 |
| Treatment - baseline | +0.250 | +0.250 | +0.250 | -0.500 | +0.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 13.250 | 10.000 | 9.500 | 13.000 | 14.500 | 4.750 |
| Treatment | 13.250 | 9.750 | 8.750 | 12.250 | 16.000 | 4.500 |
| Treatment - baseline | +0.000 | -0.250 | -0.750 | -0.750 | +1.500 | -0.250 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `b927a1b0154c199e102f249a2e54c24e4d286c45c3da3d6eff27116d87428155`
- Executable digest: `53d1cbe478951263af2133ce4a22788770d31ec601e22edf30715ddfa70d4237`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "late-conservative-base-policy-improvement-compare": {
    "first_seed": 27899,
    "games": 1,
    "output": "docs/v2/reports/late-conservative-base-policy-improvement-v1-t5-r8-c90-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5
  }
}
```
