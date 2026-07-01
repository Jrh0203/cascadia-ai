# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `terminal-policy-improvement-v1-r8-k8-h6-b8-m4`
- Games: 3 (12 seat scores per strategy)
- Baseline mean: 93.500
- Treatment mean: 94.833
- Baseline P10 / P50 / P90: 89.0 / 95.5 / 97.8
- Treatment P10 / P50 / P90: 92.0 / 94.5 / 98.0
- Baseline seat SD / range: 3.989 / 88.0-99.0
- Treatment seat SD / range: 2.855 / 91.0-100.0
- Paired delta: **+1.333**
- 95% CI: [-2.249, +4.916]
- Paired SD / SE: 3.166 / 1.828
- Game wins / ties / losses: 2 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 2.08 / 1.50 / 3.66 / 8.05 / 51.29 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 2323.46 / 2246.69 / 3988.41 / 5959.80 / 6762.05 ms
- Baseline runtime: 0.500s (0.167s/game)
- Treatment runtime: 557.633s (185.878s/game)
- Combined wall time: 558.133s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.250 | 6.333 | 5.833 | 6.500 | 5.167 |
| Treatment | 5.500 | 6.583 | 6.333 | 6.583 | 5.500 |
| Treatment - baseline | +0.250 | +0.250 | +0.500 | +0.083 | +0.333 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.833 | 11.333 | 12.333 | 12.000 | 14.333 | 4.583 |
| Treatment | 11.583 | 10.167 | 13.167 | 11.083 | 14.167 | 4.167 |
| Treatment - baseline | +1.750 | -1.167 | +0.833 | -0.917 | -0.167 | -0.417 |

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
    "first_seed": 24900,
    "games": 3,
    "output": "docs/v2/reports/terminal-policy-improvement-v2-r8-pilot3.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4
  }
}
```
