# Pattern Potential Grid Selection

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Policies: 125
- Games per policy: 32 (128 seat scores)
- Seeds: 31300 through 31331
- Production baseline: `pattern-potential-v1-k8-h6-b8-m4-a100-h000-b000` = 91.992
- Selected: `pattern-potential-v1-k8-h6-b8-m4-a100-h000-b075` = 92.117
- Selected gain: **+0.125**
- Selection gate: +0.400 (failed)
- Wall time: 46.129s

## Top Policies

| Rank | Opportunity | Habitat | Bear | Mean | Gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.00 | 0.75 | 92.117 | +0.125 |
| 2 | 1.25 | 0.00 | 1.00 | 92.016 | +0.023 |
| 3 | 1.00 | 0.00 | 0.00 | 91.992 | +0.000 |
| 4 | 1.00 | 0.00 | 1.00 | 91.984 | -0.008 |
| 5 | 1.25 | 0.50 | 0.75 | 91.961 | -0.031 |
| 6 | 1.00 | 0.00 | 0.50 | 91.883 | -0.109 |
| 7 | 1.25 | 0.00 | 0.00 | 91.859 | -0.133 |
| 8 | 0.75 | 0.00 | 1.00 | 91.859 | -0.133 |
| 9 | 1.50 | 0.50 | 0.25 | 91.828 | -0.164 |
| 10 | 1.50 | 0.75 | 0.75 | 91.820 | -0.172 |
| 11 | 1.25 | 0.00 | 0.25 | 91.812 | -0.180 |
| 12 | 1.00 | 0.00 | 0.25 | 91.773 | -0.219 |
| 13 | 1.25 | 0.25 | 0.75 | 91.773 | -0.219 |
| 14 | 0.75 | 0.00 | 0.75 | 91.766 | -0.227 |
| 15 | 0.75 | 0.00 | 0.50 | 91.711 | -0.281 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `09f3736bd3d6a1bd6d838191672486a945dabc2697483ae06364845e0339cdfc`
- Executable digest: `0af69d96fd77ccedd4f6b153bd1ad20a8c8383fe7f7a63fa09807ddd4ff703fb`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "pattern-potential-sweep": {
    "first_seed": 31300,
    "games": 32,
    "output": "docs/v2/reports/pattern-potential-v1-grid125-train32.json"
  }
}
```
