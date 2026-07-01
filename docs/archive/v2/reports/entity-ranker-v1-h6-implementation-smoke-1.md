# Cascadia Benchmark

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Strategy: `mlx-ranking-v1-k2-b1`
- Games: 1 (4 seat scores)
- Seeds: 22999 through 22999
- Mean base score: **89.000**
- 95% CI: [89.000, 89.000]
- Game-mean SD / SE: 0.000 / 0.000
- Seat-score SD: 2.160
- P10 / P50 / P90: 86.9 / 89.5 / 90.7
- Min / max: 86.0 / 91.0
- Decision latency mean / P50 / P90 / P99 / max: 9.07 / 6.23 / 10.97 / 56.64 / 216.12 ms (80 decisions)
- Runtime: 0.726s (1.377 games/s)

## Mean Breakdown

| Category | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Habitat | 4.750 | 5.500 | 7.250 | 5.750 | 5.250 |
| Wildlife | Bear 10.250 | Elk 12.000 | Salmon 10.000 | Hawk 11.250 | Fox 12.750 |

Mean remaining nature tokens: 4.250

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
  "ranking-model-benchmark": {
    "bear_candidates": 1,
    "candidates": 2,
    "first_seed": 22999,
    "games": 1,
    "model_dir": "artifacts/models/entity-ranker-v1-h6",
    "output": "docs/v2/reports/entity-ranker-v1-h6-implementation-smoke-1.json",
    "run_dir": null,
    "server": ".venv/bin/cascadia-mlx-ranking-serve"
  }
}
```
