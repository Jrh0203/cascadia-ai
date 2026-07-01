# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `habitat-candidate-lookahead-v1-k2-h1-r1-d1`
- Treatment: `mlx-value-leaf-lookahead-v1-k2-h1-r1-d1`
- Games: 1 (4 seat scores per strategy)
- Baseline mean: 90.000
- Treatment mean: 85.500
- Baseline P10 / P50 / P90: 87.6 / 89.5 / 92.8
- Treatment P10 / P50 / P90: 82.9 / 86.0 / 87.7
- Baseline seat SD / range: 2.944 / 87.0-94.0
- Treatment seat SD / range: 2.646 / 82.0-88.0
- Paired delta: **-4.500**
- 95% CI: [-4.500, -4.500]
- Paired SD / SE: 0.000 / 0.000
- Game wins / ties / losses: 0 / 0 / 1
- Baseline decision latency mean / P50 / P90 / P99 / max: 71.94 / 25.16 / 182.76 / 344.24 / 350.34 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 67.98 / 30.84 / 170.09 / 242.10 / 321.72 ms
- Baseline runtime: 5.756s (5.756s/game)
- Treatment runtime: 5.439s (5.439s/game)
- Combined wall time: 11.196s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 6.250 | 5.000 | 5.750 | 6.250 | 5.250 |
| Treatment | 6.000 | 6.750 | 5.500 | 5.250 | 5.250 |
| Treatment - baseline | -0.250 | +1.750 | -0.250 | -1.000 | +0.000 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 3.000 | 12.250 | 14.250 | 11.000 | 17.000 | 4.000 |
| Treatment | 5.500 | 11.500 | 9.500 | 11.000 | 16.500 | 2.750 |
| Treatment - baseline | +2.500 | -0.750 | -4.750 | +0.000 | -0.500 | -1.250 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `1a75a576c6daeb91b2c4ae7b8a53748b566a09d6f03acf4bf53ad77baed50235`
- Executable digest: `0d047eb747300617190f96bed15a5c1c6d832ab73f929f661328134225edee15`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[{"manifest":"/Users/johnherrick/cascadia/artifacts/models/entity-value-v1-greedy256/model.json","manifest_blake3":"cd08275b4e1bb5bf28ba15672b277377b3bc6051d2d95351398912b02f36e85f","path":"/Users/johnherrick/cascadia/artifacts/models/entity-value-v1-greedy256","role":"model_dir"}]`

### Typed Configuration

```json
{
  "value-leaf-compare": {
    "candidates": 2,
    "determinizations": 1,
    "first_seed": 22799,
    "games": 1,
    "greedy_plies": 1,
    "habitat_candidates": 1,
    "model_dir": "artifacts/models/entity-value-v1-greedy256",
    "output": "docs/v2/reports/value-leaf-implementation-smoke-1.json",
    "run_dir": null,
    "server": ".venv/bin/cascadia-mlx-serve"
  }
}
```
