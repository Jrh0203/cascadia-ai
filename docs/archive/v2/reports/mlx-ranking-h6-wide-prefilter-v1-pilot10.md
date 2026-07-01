# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Treatment: `mlx-habitat-prefilter-lookahead-v1-k16-h8-a8-p14-r4-d4`
- Games: 10 (40 seat scores per strategy)
- Baseline mean: 91.625
- Treatment mean: 91.800
- Baseline P10 / P50 / P90: 88.9 / 91.5 / 95.1
- Treatment P10 / P50 / P90: 88.0 / 91.0 / 95.1
- Baseline seat SD / range: 3.102 / 83.0-99.0
- Treatment seat SD / range: 2.980 / 86.0-98.0
- Paired delta: **+0.175**
- 95% CI: [-1.322, +1.672]
- Paired SD / SE: 2.415 / 0.764
- Game wins / ties / losses: 3 / 0 / 7
- Baseline decision latency mean / P50 / P90 / P99 / max: 86.93 / 75.35 / 163.96 / 264.15 / 346.52 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 91.70 / 79.83 / 164.50 / 258.61 / 405.84 ms
- Baseline runtime: 69.557s (6.956s/game)
- Treatment runtime: 73.370s (7.337s/game)
- Combined wall time: 142.927s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.775 | 5.950 | 6.100 | 5.550 | 6.225 |
| Treatment | 5.675 | 5.825 | 6.025 | 5.950 | 5.800 |
| Treatment - baseline | -0.100 | -0.125 | -0.075 | +0.400 | -0.425 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 5.950 | 12.500 | 12.350 | 12.125 | 15.525 | 3.575 |
| Treatment | 5.675 | 12.750 | 12.050 | 12.400 | 15.500 | 4.150 |
| Treatment - baseline | -0.275 | +0.250 | -0.300 | +0.275 | -0.025 | +0.575 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `a0c1994e21baa15e14af4b8ba815bcfdbc455ee0fe4e5cb28812697961b4acaa`
- Executable digest: `4f43419cc21ec4ecdec2e72d09339ff3090a988fbabd566a53cbc05d518a2706`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[{"manifest":"/Users/johnherrick/cascadia/artifacts/models/entity-ranker-v1-h6/model.json","manifest_blake3":"6fa654a6af96fa8cf137a5b57d808c311565bbf2d61a8ebee50c2f511218c723","path":"/Users/johnherrick/cascadia/artifacts/models/entity-ranker-v1-h6","role":"model_dir"}]`

### Typed Configuration

```json
{
  "ranking-habitat-prefilter-compare": {
    "baseline_candidates": 8,
    "baseline_habitat_candidates": 6,
    "candidates": 16,
    "determinizations": 4,
    "first_seed": 22500,
    "games": 10,
    "greedy_plies": 4,
    "habitat_candidates": 8,
    "immediate_anchors": 8,
    "model_dir": "artifacts/models/entity-ranker-v1-h6",
    "output": "docs/v2/reports/mlx-ranking-h6-wide-prefilter-v1-pilot10.json",
    "prefilter_candidates": 14,
    "run_dir": null,
    "server": ".venv/bin/cascadia-mlx-ranking-serve"
  }
}
```
