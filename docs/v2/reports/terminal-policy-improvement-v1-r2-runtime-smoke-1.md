# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `terminal-policy-improvement-v1-r2-k8-h6-b8-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.250
- Treatment mean: 90.500
- Baseline P10 / P50 / P90: 91.3 / 92.0 / 93.4
- Treatment P10 / P50 / P90: 87.1 / 92.0 / 92.7
- Baseline seat SD / range: 1.258 / 91.0-94.0
- Treatment seat SD / range: 3.697 / 85.0-93.0
- Paired delta: **-1.750**
- 95% CI: [-1.750, -1.750]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 3.73 / 3.04 / 8.16 / 13.55 / 20.82 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1254.16 / 1072.75 / 2388.82 / 2846.79 / 2901.21 ms
- Baseline runtime: 0.299s (0.299s/game)
- Treatment runtime: 100.334s (100.334s/game)
- Combined wall time: 100.633s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.250 | 6.750 | 6.250 | 5.750 | 4.500 |
| Treatment | 7.000 | 6.250 | 5.500 | 4.750 | 4.750 |
| Treatment - baseline | +0.750 | -0.500 | -0.750 | -1.000 | +0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.500 | 12.750 | 11.750 | 11.750 | 12.750 | 4.250 |
| Treatment | 11.500 | 11.750 | 12.000 | 9.500 | 13.250 | 4.250 |
| Treatment - baseline | +2.000 | -1.000 | +0.250 | -2.250 | +0.500 | +0.000 |

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
    "first_seed": 24699,
    "games": 1,
    "output": "docs/v2/reports/terminal-policy-improvement-v1-r2-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4
  }
}
```
