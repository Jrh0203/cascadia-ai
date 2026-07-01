# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-terminal-policy-improvement-v1-t5-r8-k8-h6-b8-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 92.100
- Treatment mean: 93.100
- Baseline P10 / P50 / P90: 87.0 / 92.5 / 96.0
- Treatment P10 / P50 / P90: 88.0 / 94.0 / 96.0
- Baseline seat SD / range: 3.433 / 84.0-100.0
- Treatment seat SD / range: 2.925 / 86.0-98.0
- Paired delta: **+1.000**
- 95% CI: [+0.574, +1.426]
- Paired SD / SE: 0.687 / 0.217
- Game wins / ties / losses: 9 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.62 / 1.41 / 3.07 / 4.72 / 6.34 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 143.89 / 1.42 / 526.64 / 1977.26 / 3081.28 ms
- Baseline runtime: 1.304s (0.130s/game)
- Treatment runtime: 115.123s (11.512s/game)
- Combined wall time: 116.427s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.800 | 5.875 | 6.025 | 5.700 | 5.950 |
| Treatment | 5.800 | 6.050 | 5.950 | 5.825 | 6.025 |
| Treatment - baseline | +0.000 | +0.175 | -0.075 | +0.125 | +0.075 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.850 | 12.350 | 11.600 | 12.225 | 14.800 | 3.925 |
| Treatment | 8.600 | 12.625 | 11.800 | 12.075 | 14.425 | 3.925 |
| Treatment - baseline | +0.750 | +0.275 | +0.200 | -0.150 | -0.375 | +0.000 |

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
    "first_seed": 26900,
    "games": 10,
    "output": "docs/v2/reports/late-terminal-policy-improvement-v1-t5-r8-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5
  }
}
```
