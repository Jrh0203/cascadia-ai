# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 91.580
- Treatment mean: 92.100
- Baseline P10 / P50 / P90: 87.0 / 91.0 / 96.0
- Treatment P10 / P50 / P90: 88.0 / 92.0 / 97.0
- Baseline seat SD / range: 3.339 / 84.0-100.0
- Treatment seat SD / range: 3.275 / 85.0-101.0
- Paired delta: **+0.520**
- 95% CI: [+0.260, +0.780]
- Paired SD / SE: 0.938 / 0.133
- Game wins / ties / losses: 31 / 10 / 9
- Baseline decision latency mean / P50 / P90 / P99 / max: 0.53 / 0.52 / 0.84 / 1.16 / 1.61 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 38.28 / 0.55 / 172.58 / 372.21 / 491.15 ms
- Baseline runtime: 2.146s (0.043s/game)
- Treatment runtime: 153.127s (3.063s/game)
- Combined wall time: 155.273s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.535 | 5.940 | 5.825 | 5.800 | 5.630 |
| Treatment | 5.530 | 5.980 | 5.915 | 5.935 | 5.735 |
| Treatment - baseline | -0.005 | +0.040 | +0.090 | +0.135 | +0.105 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.730 | 11.505 | 12.375 | 12.585 | 14.930 | 3.725 |
| Treatment | 8.190 | 11.415 | 12.315 | 12.475 | 14.815 | 3.795 |
| Treatment - baseline | +0.460 | -0.090 | -0.060 | -0.110 | -0.115 | +0.070 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `940109c03d88769406b6b1afb9867142e84e7c3c9defc19798699293d411def9`
- V2 source digest: `aa6e140bbdc8421ae8358d2a80b9da56ac23066e3249bff996ee2aaccd66afc5`
- Executable digest: `f811b8a0faab4ebc6add631eea12e75203da0a98189bfefd7e88c0dd71d9d2c8`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "late-conservative-base-policy-improvement-compare": {
    "first_seed": 35100,
    "games": 50,
    "output": "docs/v2/reports/canonical-redetermination-strong-requalify50.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5
  }
}
```
