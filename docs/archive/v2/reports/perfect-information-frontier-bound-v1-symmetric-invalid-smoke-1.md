# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Treatment: `perfect-information-pattern-oracle-v1-k8-h6-b8-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 93.000
- Treatment mean: 89.000
- Baseline P10 / P50 / P90: 90.9 / 93.5 / 94.7
- Treatment P10 / P50 / P90: 85.9 / 89.5 / 91.7
- Baseline seat SD / range: 2.160 / 90.0-95.0
- Treatment seat SD / range: 3.162 / 85.0-92.0
- Paired delta: **-4.000**
- 95% CI: [-4.000, -4.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 128.77 / 0.79 / 546.60 / 980.51 / 1185.54 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 122.78 / 122.29 / 191.56 / 310.86 / 369.64 ms
- Baseline runtime: 10.302s (10.302s/game)
- Treatment runtime: 9.823s (9.823s/game)
- Combined wall time: 20.126s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.750 | 6.250 | 5.750 | 6.750 | 5.250 |
| Treatment | 7.000 | 6.000 | 4.750 | 5.500 | 5.750 |
| Treatment - baseline | +1.250 | -0.250 | -1.000 | -1.250 | +0.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 12.500 | 11.250 | 11.750 | 9.500 | 14.000 | 4.250 |
| Treatment | 9.500 | 10.750 | 13.500 | 10.250 | 12.750 | 3.250 |
| Treatment - baseline | -3.000 | -0.500 | +1.750 | +0.750 | -1.250 | -1.000 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `781f2bf9add266d5ac0e9282148308871a52d2b9fcd9e9418aba877972534314`
- Executable digest: `db60ad96c4378d415bb212763d65ed3e1c28645c77578343e80872d38d607a7d`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-oracle-compare": {
    "first_seed": 28899,
    "games": 1,
    "output": "docs/v2/reports/perfect-information-frontier-bound-v1-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4
  }
}
```
