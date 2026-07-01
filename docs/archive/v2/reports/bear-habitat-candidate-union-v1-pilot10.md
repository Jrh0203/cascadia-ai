# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Treatment: `bear-habitat-candidate-lookahead-v1-k8-h6-b8-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.700
- Treatment mean: 91.400
- Baseline P10 / P50 / P90: 87.9 / 92.0 / 95.0
- Treatment P10 / P50 / P90: 88.9 / 92.0 / 96.0
- Baseline seat SD / range: 3.268 / 85.0-100.0
- Treatment seat SD / range: 2.863 / 84.0-97.0
- Paired delta: **-0.300**
- 95% CI: [-2.048, +1.448]
- Paired SD / SE: 2.821 / 0.892
- Game wins / ties / losses: 5 / 0 / 5
- Baseline decision latency mean / P50 / P90 / P99 / max: 81.01 / 70.50 / 157.40 / 239.24 / 348.57 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 101.81 / 83.75 / 191.82 / 383.30 / 798.64 ms
- Baseline runtime: 64.822s (6.482s/game)
- Treatment runtime: 81.460s (8.146s/game)
- Combined wall time: 146.281s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.750 | 5.625 | 6.075 | 5.675 | 6.100 |
| Treatment | 5.675 | 6.175 | 5.950 | 5.600 | 5.900 |
| Treatment - baseline | -0.075 | +0.550 | -0.125 | -0.075 | -0.200 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 7.400 | 12.250 | 12.450 | 12.575 | 14.475 | 3.325 |
| Treatment | 9.475 | 11.825 | 12.350 | 12.050 | 13.025 | 3.375 |
| Treatment - baseline | +2.075 | -0.425 | -0.100 | -0.525 | -1.450 | +0.050 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `a0c1994e21baa15e14af4b8ba815bcfdbc455ee0fe4e5cb28812697961b4acaa`
- Executable digest: `4f43419cc21ec4ecdec2e72d09339ff3090a988fbabd566a53cbc05d518a2706`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "bear-habitat-candidate-compare": {
    "bear_candidates": 8,
    "candidates": 8,
    "determinizations": 4,
    "first_seed": 22800,
    "games": 10,
    "greedy_plies": 4,
    "habitat_candidates": 6,
    "output": "docs/v2/reports/bear-habitat-candidate-union-v1-pilot10.json"
  }
}
```
