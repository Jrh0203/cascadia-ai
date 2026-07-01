# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `public-focal-open-loop-tree-v1-t5-s128-r16-u2000-k8-h6-b8-w2-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 92.350
- Treatment mean: 92.375
- Baseline P10 / P50 / P90: 89.0 / 92.5 / 96.0
- Treatment P10 / P50 / P90: 88.0 / 92.0 / 96.0
- Baseline seat SD / range: 2.656 / 87.0-97.0
- Treatment seat SD / range: 2.897 / 86.0-97.0
- Paired delta: **+0.025**
- 95% CI: [-0.856, +0.906]
- Paired SD / SE: 1.421 / 0.449
- Game wins / ties / losses: 6 / 0 / 4
- Baseline decision latency mean / P50 / P90 / P99 / max: 42.14 / 0.53 / 172.60 / 432.55 / 480.91 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 184.77 / 0.53 / 897.85 / 1734.40 / 1952.69 ms
- Baseline runtime: 33.715s (3.371s/game)
- Treatment runtime: 147.820s (14.782s/game)
- Combined wall time: 181.535s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.600 | 5.425 | 5.625 | 5.775 | 6.150 |
| Treatment | 5.600 | 5.750 | 5.750 | 5.775 | 5.900 |
| Treatment - baseline | +0.000 | +0.325 | +0.125 | +0.000 | -0.250 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.300 | 12.500 | 12.400 | 11.625 | 13.825 | 4.125 |
| Treatment | 8.750 | 12.050 | 12.250 | 11.875 | 14.600 | 4.075 |
| Treatment - baseline | -0.550 | -0.450 | -0.150 | +0.250 | +0.775 | -0.050 |

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
  "public-focal-tree-compare": {
    "exploration_milli": 2000,
    "first_seed": 34900,
    "games": 10,
    "output": "docs/v2/reports/public-focal-open-loop-tree-v1-pilot10.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "root_candidates": 16,
    "sequential": true,
    "simulations": 128,
    "terminal_turns": 5,
    "wildlife_candidates": 2
  }
}
```
