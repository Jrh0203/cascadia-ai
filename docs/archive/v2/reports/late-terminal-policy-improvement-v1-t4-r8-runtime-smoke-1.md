# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-terminal-policy-improvement-v1-t4-r8-k8-h6-b8-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.750
- Treatment mean: 93.750
- Baseline P10 / P50 / P90: 92.0 / 92.5 / 93.7
- Treatment P10 / P50 / P90: 92.6 / 94.0 / 94.7
- Baseline seat SD / range: 0.957 / 92.0-94.0
- Treatment seat SD / range: 1.258 / 92.0-95.0
- Paired delta: **+1.000**
- 95% CI: [+1.000, +1.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.08 / 0.82 / 2.04 / 3.91 / 4.90 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 65.89 / 0.83 / 261.65 / 769.35 / 986.56 ms
- Baseline runtime: 0.087s (0.087s/game)
- Treatment runtime: 5.272s (5.272s/game)
- Combined wall time: 5.358s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 4.500 | 5.500 | 6.250 | 5.750 | 6.500 |
| Treatment | 5.000 | 5.750 | 5.500 | 5.250 | 7.000 |
| Treatment - baseline | +0.500 | +0.250 | -0.750 | -0.500 | +0.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.500 | 10.750 | 10.250 | 14.750 | 16.250 | 3.750 |
| Treatment | 10.250 | 10.000 | 10.750 | 13.750 | 16.750 | 3.750 |
| Treatment - baseline | +1.750 | -0.750 | +0.500 | -1.000 | +0.500 | +0.000 |

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
    "first_seed": 26599,
    "games": 1,
    "output": "docs/v2/reports/late-terminal-policy-improvement-v1-t4-r8-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 4
  }
}
```
