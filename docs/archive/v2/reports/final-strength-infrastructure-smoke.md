# Final Cascadia V2 Strength Validation

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Games: 1 (4 treatment seat scores)
- Seeds: 999999 through 999999
- Treatment: `canonical-action-legacy-exact-mlx-v1-k32-r600-lmr-no-paid-prelude`
- Mean base score: **94.500**
- Game-block 95% CI: [94.500, 94.500]
- 100-point target: **not reached**

## Paired Canonical V2 Control

- Baseline: `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
- Baseline mean: 94.750
- Paired delta: -0.250
- Paired 95% CI: [-0.250, -0.250]
- Record: 0-0-1

## Score Distribution

| Metric | Value |
|---|---:|
| Game-mean SD | 0.000 |
| Seat-score SD | 4.796 |
| Standard error | 0.000 |
| P10 | 90.6 |
| P50 | 93.5 |
| P90 | 99.2 |

## Integrity

- Complete held-out suite: `True`
- All one-game smoke gates passed: `True`
- All MLX services shut down cleanly: `True`
- Distinct hosts: john3
- Source revisions: 6b4a43a95df1b6fbec9160fbe8ceb664be69cbee

## Independent V1 Reference

- Reproduced v1 mean: 95.895 over 50 games
- Absolute treatment-minus-v1 difference: -1.395
- This is an absolute cross-engine reference, not a paired canonical comparison.
