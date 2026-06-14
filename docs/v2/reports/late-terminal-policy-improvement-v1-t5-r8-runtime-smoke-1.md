# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-terminal-policy-improvement-v1-t5-r8-k8-h6-b8-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.000
- Treatment mean: 90.250
- Baseline P10 / P50 / P90: 89.5 / 93.0 / 93.7
- Treatment P10 / P50 / P90: 88.3 / 90.0 / 92.4
- Baseline seat SD / range: 2.708 / 88.0-94.0
- Treatment seat SD / range: 2.217 / 88.0-93.0
- Paired delta: **-1.750**
- 95% CI: [-1.750, -1.750]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.00 / 0.93 / 1.72 / 2.24 / 2.38 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 165.00 / 0.94 / 720.06 / 1407.07 / 2225.82 ms
- Baseline runtime: 0.080s (0.080s/game)
- Treatment runtime: 13.201s (13.201s/game)
- Combined wall time: 13.281s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.750 | 6.500 | 6.000 | 5.250 | 5.250 |
| Treatment | 5.000 | 6.250 | 5.000 | 5.500 | 6.000 |
| Treatment - baseline | -0.750 | -0.250 | -1.000 | +0.250 | +0.750 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.250 | 12.250 | 10.500 | 10.250 | 17.250 | 3.750 |
| Treatment | 7.500 | 12.250 | 10.750 | 10.250 | 17.500 | 4.250 |
| Treatment - baseline | -1.750 | +0.000 | +0.250 | +0.000 | +0.250 | +0.500 |

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
    "first_seed": 26899,
    "games": 1,
    "output": "docs/v2/reports/late-terminal-policy-improvement-v1-t5-r8-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5
  }
}
```
