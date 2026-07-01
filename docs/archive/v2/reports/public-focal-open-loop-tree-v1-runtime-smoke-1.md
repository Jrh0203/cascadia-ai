# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `public-focal-open-loop-tree-v1-t5-s128-r16-u2000-k8-h6-b8-w2-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.500
- Treatment mean: 93.000
- Baseline P10 / P50 / P90: 88.4 / 94.0 / 95.4
- Treatment P10 / P50 / P90: 87.7 / 94.5 / 97.1
- Baseline seat SD / range: 4.435 / 86.0-96.0
- Treatment seat SD / range: 5.598 / 85.0-98.0
- Paired delta: **+0.500**
- 95% CI: [+0.500, +0.500]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 1 / 0 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 36.47 / 0.49 / 178.30 / 248.47 / 347.07 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 148.32 / 0.49 / 689.06 / 1334.80 / 1391.79 ms
- Baseline runtime: 2.918s (2.918s/game)
- Treatment runtime: 11.866s (11.866s/game)
- Combined wall time: 14.784s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.500 | 6.500 | 5.000 | 5.500 | 7.000 |
| Treatment | 5.750 | 6.000 | 5.750 | 5.500 | 6.000 |
| Treatment - baseline | +0.250 | -0.500 | +0.750 | +0.000 | -1.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.250 | 12.000 | 14.250 | 13.250 | 11.750 | 3.500 |
| Treatment | 8.500 | 11.750 | 13.750 | 12.000 | 13.750 | 4.250 |
| Treatment - baseline | +0.250 | -0.250 | -0.500 | -1.250 | +2.000 | +0.750 |

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
    "first_seed": 34899,
    "games": 1,
    "output": "docs/v2/reports/public-focal-open-loop-tree-v1-runtime-smoke-1.json",
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
