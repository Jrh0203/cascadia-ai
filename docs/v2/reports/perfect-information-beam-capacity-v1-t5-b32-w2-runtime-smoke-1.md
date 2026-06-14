# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Treatment: `perfect-information-focal-beam-v1-t5-b32-k8-h6-b8-w2-m4`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 91.500
- Treatment mean: 91.500
- Baseline P10 / P50 / P90: 89.2 / 92.5 / 93.0
- Treatment P10 / P50 / P90: 89.2 / 92.5 / 93.0
- Baseline seat SD / range: 2.380 / 88.0-93.0
- Treatment seat SD / range: 2.380 / 88.0-93.0
- Paired delta: **+0.000**
- 95% CI: [+0.000, +0.000]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 1 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 960.52 / 117.99 / 2228.96 / 12676.57 / 16001.41 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1136.86 / 105.83 / 3181.63 / 15938.34 / 18242.20 ms
- Baseline runtime: 76.984s (76.984s/game)
- Treatment runtime: 91.069s (91.069s/game)
- Combined wall time: 168.052s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 4.750 | 5.500 | 5.750 | 4.750 | 8.000 |
| Treatment | 4.750 | 5.500 | 5.750 | 4.750 | 8.000 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 19.000 | 5.000 | 13.000 | 9.500 | 14.000 | 2.250 |
| Treatment | 19.000 | 5.000 | 13.000 | 9.500 | 14.000 | 2.250 |
| Treatment - baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `fcdef7d4ba1ebfa12744d8863993da25df4e12a8ece8d4abf25e98546b567163`
- Executable digest: `78f3eecb83d62eb49a9ed7f4828aef886ca377037a29292133121132dd15348f`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "perfect-information-beam-capacity-compare": {
    "baseline_beam_width": 16,
    "first_seed": 30899,
    "games": 1,
    "output": "docs/v2/reports/perfect-information-beam-capacity-v1-t5-b32-w2-runtime-smoke-1.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "terminal_turns": 5,
    "treatment_beam_width": 32,
    "wildlife_candidates": 2
  }
}
```
