# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `perfect-information-focal-beam-v1-t5-b16-k8-h6-b8-w2-m4`
- Treatment: `perfect-information-focal-beam-v1-t5-b32-k8-h6-b8-w2-m4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 94.075
- Treatment mean: 94.100
- Baseline P10 / P50 / P90: 88.0 / 95.0 / 97.1
- Treatment P10 / P50 / P90: 88.0 / 95.0 / 97.1
- Baseline seat SD / range: 3.377 / 87.0-102.0
- Treatment seat SD / range: 3.395 / 87.0-102.0
- Paired delta: **+0.025**
- 95% CI: [-0.024, +0.074]
- Paired SD / SE: 0.079 / 0.025
- Game wins / ties / losses: 1 / 9 / 0
- Baseline decision latency mean / P50 / P90 / P99 / max: 821.36 / 116.87 / 2465.77 / 10453.13 / 16756.62 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 1192.66 / 115.68 / 3055.67 / 17188.93 / 25422.11 ms
- Baseline runtime: 658.489s (65.849s/game)
- Treatment runtime: 955.477s (95.548s/game)
- Combined wall time: 1613.966s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.450 | 6.250 | 5.825 | 5.900 | 5.975 |
| Treatment | 5.500 | 6.200 | 5.825 | 5.900 | 5.975 |
| Treatment - baseline | +0.050 | -0.050 | +0.000 | +0.000 | +0.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 17.900 | 6.825 | 11.150 | 9.100 | 15.700 | 4.000 |
| Treatment | 17.900 | 6.825 | 11.075 | 9.100 | 15.800 | 4.000 |
| Treatment - baseline | +0.000 | +0.000 | -0.075 | +0.000 | +0.100 | +0.000 |

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
    "first_seed": 30900,
    "games": 10,
    "output": "docs/v2/reports/perfect-information-beam-capacity-v1-t5-b32-w2-pilot10.json",
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
