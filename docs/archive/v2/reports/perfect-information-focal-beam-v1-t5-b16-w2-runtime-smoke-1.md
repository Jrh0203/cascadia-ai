# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-pattern-oracle-v1-k8-h6-b8-w2-m4`
- Treatment: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 95.250
- Treatment mean: 95.250
- Baseline P10 / P50 / P90: 92.9 / 95.5 / 97.4
- Treatment P10 / P50 / P90: 92.9 / 95.5 / 97.4
- Baseline seat SD / range: 2.500 / 92.0-98.0
- Treatment seat SD / range: 2.500 / 92.0-98.0
- Paired delta: **+0.000**
- 95% CI: [+0.000, +0.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 1 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 247.44 / 257.18 / 404.53 / 532.26 / 582.43 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 2043.80 / 276.78 / 5782.69 / 23913.12 / 31543.99 ms
- Baseline runtime: 20.225s (20.225s/game)
- Treatment runtime: 163.852s (163.852s/game)
- Combined wall time: 184.077s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.500 | 7.500 | 5.000 | 4.750 | 6.750 |
| Treatment | 6.500 | 7.500 | 5.000 | 4.750 | 6.750 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 14.250 | 7.000 | 12.000 | 7.250 | 20.500 | 3.750 |
| Treatment | 14.250 | 7.000 | 12.000 | 7.250 | 20.500 | 3.750 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `9a2ac488021a979609cc7396c4d9418810b3cedf9b679c32aa0d9cd53952e6c5`
- Executable digest: `00e617be37ab4cff9b8b8d5efb0e0ed8f1327880a632dabe22409d9af0e4a96a`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-focal-beam-compare": {
    "beam_width": 16,
    "first_seed": 29599,
    "games": 1,
    "output": "docs/v2/reports/perfect-information-focal-beam-v1-t5-b16-w2-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "terminal_turns": 5,
    "wildlife_candidates": 2
  }
}
```
