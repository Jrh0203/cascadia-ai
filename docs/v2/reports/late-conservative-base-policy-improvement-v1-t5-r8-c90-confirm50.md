# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 91.495
- Treatment mean: 91.915
- Baseline P10 / P50 / P90: 88.0 / 91.0 / 95.0
- Treatment P10 / P50 / P90: 88.9 / 92.0 / 96.0
- Baseline seat SD / range: 3.113 / 82.0-101.0
- Treatment seat SD / range: 3.000 / 82.0-101.0
- Paired delta: **+0.420**
- 95% CI: [+0.179, +0.661]
- Paired SD / SE: 0.871 / 0.123
- Game wins / ties / losses: 28 / 9 / 13
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.26 / 1.21 / 2.11 / 3.05 / 6.83 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 87.43 / 1.20 / 362.24 / 880.54 / 2252.81 ms
- Baseline runtime: 5.077s (0.102s/game)
- Treatment runtime: 349.739s (6.995s/game)
- Combined wall time: 354.816s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.660 | 5.640 | 5.975 | 5.765 | 5.580 |
| Treatment | 5.645 | 5.770 | 6.000 | 5.900 | 5.670 |
| Treatment - baseline | -0.015 | +0.130 | +0.025 | +0.135 | +0.090 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.335 | 11.740 | 12.510 | 12.200 | 14.300 | 3.790 |
| Treatment | 8.415 | 11.845 | 12.505 | 12.245 | 14.190 | 3.730 |
| Treatment - baseline | +0.080 | +0.105 | -0.005 | +0.045 | -0.110 | -0.060 |

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
    "first_seed": 28000,
    "games": 50,
    "output": "docs/v2/reports/late-conservative-base-policy-improvement-v1-t5-r8-c90-confirm50.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5
  }
}
```
