# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Treatment: `mlx-habitat-rollout-lookahead-v1-k8-h6-r4-d4-rk8-rh6`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 92.250
- Treatment mean: 88.750
- Baseline P10 / P50 / P90: 89.2 / 93.0 / 94.7
- Treatment P10 / P50 / P90: 83.2 / 89.5 / 93.7
- Baseline seat SD / range: 3.096 / 88.0-95.0
- Treatment seat SD / range: 5.737 / 82.0-94.0
- Paired delta: **-3.500**
- 95% CI: [-3.500, -3.500]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 93.38 / 86.58 / 152.35 / 229.29 / 231.49 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 2423.29 / 1939.91 / 4096.78 / 8737.18 / 10405.07 ms
- Baseline runtime: 7.472s (7.472s/game)
- Treatment runtime: 196.022s (196.022s/game)
- Combined wall time: 203.501s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 7.500 | 5.250 | 5.250 | 5.000 | 6.250 |
| Treatment | 4.750 | 5.500 | 5.250 | 6.750 | 5.250 |
| Treatment - baseline | -2.750 | +0.250 | +0.000 | +1.750 | -1.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 3.750 | 14.750 | 11.500 | 12.000 | 17.000 | 4.000 |
| Treatment | 5.750 | 13.750 | 9.250 | 11.250 | 19.250 | 2.000 |
| Treatment - baseline | +2.000 | -1.000 | -2.250 | -0.750 | +2.250 | -2.000 |

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
  "ranking-habitat-rollout-compare": {
    "candidates": 8,
    "determinizations": 4,
    "first_seed": 22699,
    "games": 1,
    "habitat_candidates": 6,
    "model_dir": "artifacts/models/entity-ranker-v1-h6",
    "output": "docs/v2/reports/mlx-ranking-h6-rollout-policy-v1-runtime-smoke-1.json",
    "rollout_candidates": 8,
    "rollout_habitat_candidates": 6,
    "rollout_plies": 4,
    "run_dir": null,
    "server": ".venv/bin/cascadia-mlx-ranking-serve"
  }
}
```
