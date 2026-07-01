# Final Cascadia V2 Strength Validation

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Games: 1000 (4000 treatment seat scores)
- Seeds: 0 through 999
- Treatment: `canonical-action-legacy-exact-mlx-v1-k32-r600-lmr-no-paid-prelude`
- Mean base score: **95.744**
- Game-block 95% CI: [95.652, 95.837]
- 100-point target: **not reached**
- Target shortfall: 4.256

## Paired Canonical V2 Control

- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Baseline mean: 92.118
- Paired delta: +3.627
- Paired 95% CI: [+3.496, +3.757]
- Record: 944-10-46

## Score Distribution

| Metric | Value |
|---|---:|
| Game-mean SD | 1.494 |
| Seat-score SD | 2.958 |
| Standard error | 0.047 |
| P10 | 92.0 |
| P50 | 96.0 |
| P90 | 99.0 |

## Mean Score Breakdown

| Component | Treatment | Baseline | Delta |
|---|---:|---:|---:|
| Habitat | 30.878 | 28.919 | +1.959 |
| Bear | 11.363 | 8.369 | +2.994 |
| Elk | 10.546 | 11.808 | -1.262 |
| Salmon | 12.774 | 12.315 | +0.459 |
| Hawk | 11.380 | 12.195 | -0.815 |
| Fox | 14.920 | 14.672 | +0.248 |
| Nature Tokens | 3.883 | 3.840 | +0.043 |
| Base total | 95.744 | 92.118 | +3.627 |

## Decision Latency

| Metric | Treatment | Baseline |
|---|---:|---:|
| Mean decision | 1751.4 ms | 36.7 ms |
| P50 decision | 1863.8 ms | 0.5 ms |
| P90 decision | 2907.1 ms | 164.0 ms |
| P99 decision | 3254.2 ms | 359.2 ms |
| Seconds per game | 140.110 | 2.934 |

## Integrity

- Complete held-out suite: `True`
- All one-game smoke gates passed: `True`
- All MLX services shut down cleanly: `True`
- Host allocation: Johns-Mac-mini.local=334, john2=333, john3=333
- Source revisions: cb7225e8d10167153fa681fef33d8e5ce491c0a2
- Binary SHA256: `613d45350c6daf4e3b4122f7e14cbc9ab957d6c8ef7055846bd1da297b6ebb83`
- MLX model SHA256: `9fd11f704a5feb427aab324c19dc819213dda08f8a4b90331999df3726b11f89`
- Legacy weights SHA256: `f40627623d3686d7d2d6a2f8f109445f54e449f0d7045552ebe831f955a58f48`

## Independent V1 Reference

- Reproduced v1 mean: 95.895 over 50 games
- V1 game-block 95% CI: [95.480, 96.310]
- Absolute treatment-minus-v1 difference: -0.151
- This is an absolute cross-engine reference, not a paired canonical comparison.
