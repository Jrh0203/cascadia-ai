# Exact MLX Rollout-Budget Pilot

Experiment: `exact-mlx-rollout-budget-r600-r1200-pilot-v1-20260612`

## Result

Rejected. Doubling sequential-halving budget did not clear the +0.50 pilot
gate.

| Metric | R600 | R1200 | Delta |
|---|---:|---:|---:|
| Mean base score | 97.583 | 97.750 | +0.167 |
| Seconds/game | 151.19 | 311.43 | +160.24 |
| Neural rows | 18,866,076 | 39,140,824 | +20,274,748 |

Paired 95% CI: `[-2.962,+3.296]`. Record: 2 wins, zero ties, one loss.
Wildlife changed by +0.750, habitat by -0.083, and Nature Tokens by -0.500.

Both arms completed every canonical action legally with zero bridge or neural
fallback, and both services shut down cleanly. The null result is therefore a
search-strength finding, not an infrastructure failure.

Machine-readable report: `exact-mlx-rollout-budget-r600-r1200-pilot3.json`

BLAKE3:
`5a81d4ebbc2a49269316f771c3729c30f068f24bc777289ef2c81890489166ed`
