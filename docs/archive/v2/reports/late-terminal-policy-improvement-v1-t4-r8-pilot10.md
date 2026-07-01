# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-terminal-policy-improvement-v1-t4-r8-k8-h6-b8-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.775
- Treatment mean: 92.250
- Baseline P10 / P50 / P90: 89.0 / 92.0 / 95.0
- Treatment P10 / P50 / P90: 89.0 / 92.0 / 96.0
- Baseline seat SD / range: 2.423 / 87.0-97.0
- Treatment seat SD / range: 2.667 / 84.0-97.0
- Paired delta: **+0.475**
- 95% CI: [+0.197, +0.753]
- Paired SD / SE: 0.448 / 0.142
- Game wins / ties / losses: 8 / 1 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.34 / 1.18 / 2.38 / 4.25 / 8.98 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 73.68 / 1.13 / 315.05 / 995.07 / 1818.22 ms
- Baseline runtime: 1.079s (0.108s/game)
- Treatment runtime: 58.947s (5.895s/game)
- Combined wall time: 60.027s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.850 | 5.575 | 5.775 | 5.975 | 5.875 |
| Treatment | 5.875 | 6.000 | 5.625 | 5.975 | 5.800 |
| Treatment - baseline | +0.025 | +0.425 | -0.150 | +0.000 | -0.075 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.525 | 11.850 | 12.200 | 11.650 | 14.725 | 3.775 |
| Treatment | 8.425 | 12.050 | 11.975 | 12.050 | 14.675 | 3.800 |
| Treatment - baseline | -0.100 | +0.200 | -0.225 | +0.400 | -0.050 | +0.025 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `70d26870fcbdaecf38940bf7e4a26b7d64444b801593db2fa1b1d335da4aa09b`
- Executable digest: `59ac16c48fd95700599f586bd99f0c474d284b117c784c2b6d9793ff1b174fc8`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "late-terminal-policy-improvement-compare": {
    "determinizations": 8,
    "first_seed": 26600,
    "games": 10,
    "output": "docs/v2/reports/late-terminal-policy-improvement-v1-t4-r8-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 4
  }
}
```
