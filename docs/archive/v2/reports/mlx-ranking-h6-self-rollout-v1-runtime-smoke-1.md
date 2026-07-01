# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Treatment: `mlx-self-rollout-lookahead-v1-k8-h6-r4-d4-pk8-ph6`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 91.000
- Treatment mean: 88.500
- Baseline P10 / P50 / P90: 88.3 / 91.0 / 93.7
- Treatment P10 / P50 / P90: 83.0 / 91.0 / 92.0
- Baseline seat SD / range: 2.944 / 88.0-94.0
- Treatment seat SD / range: 5.745 / 80.0-92.0
- Paired delta: **-2.500**
- 95% CI: [-2.500, -2.500]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 108.12 / 111.32 / 178.00 / 219.76 / 228.54 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 782.20 / 605.22 / 1566.64 / 2645.18 / 5525.45 ms
- Baseline runtime: 8.651s (8.651s/game)
- Treatment runtime: 62.582s (62.582s/game)
- Combined wall time: 71.235s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.500 | 5.000 | 7.250 | 5.250 | 4.750 |
| Treatment | 5.750 | 5.750 | 6.000 | 5.750 | 6.250 |
| Treatment - baseline | -0.750 | +0.750 | -1.250 | +0.500 | +1.500 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 6.750 | 11.250 | 13.000 | 14.500 | 13.750 | 3.000 |
| Treatment | 5.500 | 13.000 | 13.500 | 14.500 | 9.500 | 3.000 |
| Treatment - baseline | -1.250 | +1.750 | +0.500 | +0.000 | -4.250 | +0.000 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `8aad4c7d1509fc3236ea6e885b960a4dd1d73d892c60be3939bc97d0bcac7e02`
- Executable digest: `9d8163704541737c545e6dea7dabab1fb869befbf9d17e253d3e3a3f9248a5d3`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[{"manifest":"/Users/johnherrick/cascadia/artifacts/models/entity-ranker-v1-h6/model.json","manifest_blake3":"6fa654a6af96fa8cf137a5b57d808c311565bbf2d61a8ebee50c2f511218c723","path":"/Users/johnherrick/cascadia/artifacts/models/entity-ranker-v1-h6","role":"model_dir"}]`

### Typed Configuration

```json
{
  "ranking-self-rollout-compare": {
    "candidates": 8,
    "determinizations": 4,
    "first_seed": 22999,
    "games": 1,
    "habitat_candidates": 6,
    "model_dir": "artifacts/models/entity-ranker-v1-h6",
    "output": "docs/v2/reports/mlx-ranking-h6-self-rollout-v1-runtime-smoke-1.json",
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "rollout_plies": 4,
    "run_dir": null,
    "server": ".venv/bin/cascadia-mlx-ranking-serve"
  }
}
```
