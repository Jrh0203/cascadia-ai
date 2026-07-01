# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `terminal-policy-improvement-v1-r8-k8-h6-b8-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 91.250
- Treatment mean: 93.500
- Baseline P10 / P50 / P90: 88.9 / 92.0 / 93.0
- Treatment P10 / P50 / P90: 88.8 / 94.5 / 97.4
- Baseline seat SD / range: 2.363 / 88.0-93.0
- Treatment seat SD / range: 4.796 / 87.0-98.0
- Paired delta: **+2.250**
- 95% CI: [+2.250, +2.250]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.81 / 1.42 / 3.45 / 5.15 / 5.42 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 3423.55 / 3122.46 / 6885.11 / 9978.82 / 11085.62 ms
- Baseline runtime: 0.145s (0.145s/game)
- Treatment runtime: 273.884s (273.884s/game)
- Combined wall time: 274.029s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.000 | 6.000 | 6.000 | 6.250 | 6.250 |
| Treatment | 5.500 | 5.250 | 6.500 | 5.750 | 6.500 |
| Treatment - baseline | +0.500 | -0.750 | +0.500 | -0.500 | +0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.500 | 13.500 | 9.500 | 12.500 | 12.750 | 4.000 |
| Treatment | 11.000 | 9.000 | 13.000 | 11.000 | 15.000 | 5.000 |
| Treatment - baseline | +1.500 | -4.500 | +3.500 | -1.500 | +2.250 | +1.000 |

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
    "determinizations": 8,
    "first_seed": 24899,
    "games": 1,
    "output": "docs/v2/reports/terminal-policy-improvement-v2-r8-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4
  }
}
```
