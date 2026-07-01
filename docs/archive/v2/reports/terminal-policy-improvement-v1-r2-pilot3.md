# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `terminal-policy-improvement-v1-r2-k8-h6-b8-m4`
- Games: 3 (12 seat scores per strategy)
- Baseline mean: 91.917
- Treatment mean: 92.167
- Baseline P10 / P50 / P90: 88.3 / 91.5 / 95.8
- Treatment P10 / P50 / P90: 88.1 / 93.0 / 96.7
- Baseline seat SD / range: 3.288 / 85.0-97.0
- Treatment seat SD / range: 3.689 / 85.0-98.0
- Paired delta: **+0.250**
- 95% CI: [-4.458, +4.958]
- Paired SD / SE: 4.161 / 2.402
- Game wins / ties / losses: 1 / 0 / 2
- Baseline decision latency mean / P50 / P90 / P99 / max: 2.41 / 2.22 / 4.25 / 5.74 / 6.26 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 631.81 / 577.91 / 1068.36 / 3209.60 / 3543.59 ms
- Baseline runtime: 0.579s (0.193s/game)
- Treatment runtime: 151.636s (50.545s/game)
- Combined wall time: 152.215s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.833 | 5.333 | 5.083 | 6.000 | 6.083 |
| Treatment | 5.667 | 4.917 | 6.500 | 6.250 | 5.917 |
| Treatment - baseline | -0.167 | -0.417 | +1.417 | +0.250 | -0.167 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.333 | 9.917 | 11.000 | 12.667 | 15.667 | 5.000 |
| Treatment | 10.583 | 11.583 | 11.333 | 10.333 | 15.917 | 3.167 |
| Treatment - baseline | +1.250 | +1.667 | +0.333 | -2.333 | +0.250 | -1.833 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `0052d476f7cdfea5bf25044f0cad0b36c41e4df40e6337bf95bf71dac73b224d`
- Executable digest: `b938d5cf763f5544e7621e122f57aeeacd89a4f6eb9776155f92c07cfd99c30a`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "terminal-policy-improvement-compare": {
    "determinizations": 2,
    "first_seed": 24700,
    "games": 3,
    "output": "docs/v2/reports/terminal-policy-improvement-v1-r2-pilot3.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4
  }
}
```
