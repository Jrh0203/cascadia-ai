# Public Beam-State Value Observability

- Dataset: `public-beam-value-public-beam-state-value-observability-v1-r8x2-b16-w2-20260611-train-40000`
- Games / groups / candidates: 2 / 32 / 586
- Candidate-value correlation: 0.9914 (gate >= 0.60)
- Centered-advantage correlation: 0.9365 (gate >= 0.50)
- Top-action agreement: 21/32 = 65.62% (gate >= 50%)
- Mean top-action regret: 0.1133 (gate <= 0.50)
- Maximum top-action regret: 0.6250
- Mean absolute batch difference: 0.3686
- Mean absolute centered difference: 0.3018
- Mean within-group value range: 3.3965
- Mean candidate batch standard deviation: 0.7237
- Runtime: 1698.895s
- Verdict: **PASS**

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `5721a158946a62341bbde490e40ff82feb420aa5842a2adf7ed1ec11bb7a3c27`
- Executable digest: `581dd53ac444eb4c38cec816084c3ba3f067a5546a65ffea96f2b9ec050ad1d6`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "public-beam-value-probe": {
    "first_game_index": 40000,
    "games": 2,
    "output": "artifacts/datasets/public-beam-state-value-observability-v1",
    "report": "docs/v2/reports/public-beam-state-value-observability-v1-r8x2-b16-w2.json",
    "resume": true
  }
}
```
