# Cascadia Paired Comparison

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Baseline: `pattern-aware-v1-k8-h6-b8-m4`
- Treatment: `late-terminal-policy-improvement-v1-t5-r8-k8-h6-b8-m4`
- Games: 50 (200 seat scores per strategy)
- Baseline mean: 91.710
- Treatment mean: 92.135
- Baseline P10 / P50 / P90: 88.0 / 92.0 / 95.0
- Treatment P10 / P50 / P90: 89.0 / 92.0 / 96.0
- Baseline seat SD / range: 2.884 / 84.0-99.0
- Treatment seat SD / range: 2.712 / 86.0-101.0
- Paired delta: **+0.425**
- 95% CI: [+0.198, +0.652]
- Paired SD / SE: 0.819 / 0.116
- Game wins / ties / losses: 35 / 6 / 9
- Baseline decision latency mean / P50 / P90 / P99 / max: 1.28 / 1.21 / 2.12 / 3.23 / 10.77 ms
- Treatment decision latency mean / P50 / P90 / P99 / max: 94.11 / 1.23 / 382.11 / 1023.06 / 2142.01 ms
- Baseline runtime: 5.147s (0.103s/game)
- Treatment runtime: 376.499s (7.530s/game)
- Combined wall time: 381.646s

## Mean Breakdown

| Habitat | Mountain | Forest | Prairie | Wetland | River |
|---|---:|---:|---:|---:|---:|
| Baseline | 5.525 | 5.985 | 5.675 | 5.810 | 5.720 |
| Treatment | 5.630 | 5.950 | 5.780 | 5.835 | 5.860 |
| Treatment - baseline | +0.105 | -0.035 | +0.105 | +0.025 | +0.140 |

| Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.280 | 11.960 | 12.060 | 12.140 | 14.815 | 3.740 |
| Treatment | 8.480 | 12.000 | 12.075 | 12.060 | 14.785 | 3.680 |
| Treatment - baseline | +0.200 | +0.040 | +0.015 | -0.080 | -0.030 | -0.060 |

## Reproduction

- Git revision: `a9918946f66c237a803b23ea299c6a514785ae52`
- Dirty tree / status digest: true / `040649573460f9e760e60a31f702c6b4fd2d529fbf4cde3f99f4959353a08d33`
- V2 source digest: `2171c1c6d2a729f2854001e315035b8bd975ba5aa855db2699ed78e2bf504c30`
- Executable digest: `0ae12ec6d17ba44e64cc3968996b057ac0af77c862562028e604aa2479e1669b`
- Hardware: `{"architecture":"aarch64","chip":"Apple M4","logical_cpu_count":10,"memory_bytes":"17179869184","operating_system":"macOS 26.2"}`
- Toolchain: `{"cargo":"cargo 1.94.1 (29ea6fb6a 2026-03-24)","package_version":"0.1.0","rustc":"rustc 1.94.1 (e408947bf 2026-03-25)"}`
- Input artifacts: `[]`

### Typed Configuration

```json
{
  "late-terminal-policy-improvement-compare": {
    "determinizations": 8,
    "first_seed": 27000,
    "games": 50,
    "output": "docs/v2/reports/late-terminal-policy-improvement-v1-t5-r8-confirm50.json",
    "policy_bear_candidates": 8,
    "policy_candidates": 8,
    "policy_habitat_candidates": 6,
    "policy_market_draws": 4,
    "sequential": true,
    "terminal_turns": 5
  }
}
```
