# Full-Legal Public Oracle V1 Pilot

- Status: **pilot_failed**
- Games: 12
- Baseline mean: 96.208 [95.458, 96.896]
- Treatment mean: 98.583 [97.812, 99.312]
- Paired delta: **2.375 [1.375, 3.333]**
- Record: 11 wins, 0 ties, 1 losses
- Mean local champion regret: 0.275
- Action change rate: 59.583%
- Top-64 winner rate: 91.458%

## Gates

| Gate | Passed |
|---|---:|
| `all_games_and_decisions_complete` | `True` |
| `single_source_binary_model_identity` | `True` |
| `all_integrity_checks_passed` | `True` |
| `host_memory_and_swap_telemetry_complete` | `True` |
| `process_swaps_zero` | `True` |
| `treatment_mean_at_least_threshold` | `False` |
| `paired_delta_at_least_threshold` | `False` |
| `paired_delta_bootstrap_lower_bound_positive` | `True` |
| `every_host_paired_delta_nonnegative` | `True` |
| `complete_phase_coverage` | `True` |

## Hosts

| Host | Games | Baseline | Treatment | Delta |
|---|---:|---:|---:|---:|
| john1 | 4 | 96.375 | 99.875 | +3.500 |
| john2 | 4 | 94.938 | 98.062 | +3.125 |
| john3 | 4 | 97.312 | 97.812 | +0.500 |

## Phase Diagnostics

| Phase | Decisions | Mean regret | Action change |
|---|---:|---:|---:|
| Early | 336 | 0.441 | 66.369% |
| Middle | 336 | 0.262 | 57.738% |
| Late | 288 | 0.096 | 53.819% |

## Seed Pairs

| Seed | Host | Baseline | Treatment | Delta |
|---:|---|---:|---:|---:|
| 62020 | john1 | 96.500 | 99.750 | +3.250 |
| 62021 | john1 | 96.000 | 99.500 | +3.500 |
| 62022 | john1 | 95.250 | 100.500 | +5.250 |
| 62023 | john1 | 97.750 | 99.750 | +2.000 |
| 62024 | john2 | 94.750 | 96.250 | +1.500 |
| 62025 | john2 | 95.500 | 98.500 | +3.000 |
| 62026 | john2 | 93.500 | 98.000 | +4.500 |
| 62027 | john2 | 96.000 | 99.500 | +3.500 |
| 62028 | john3 | 96.750 | 97.500 | +0.750 |
| 62029 | john3 | 98.000 | 99.500 | +1.500 |
| 62030 | john3 | 97.500 | 96.250 | -1.250 |
| 62031 | john3 | 97.000 | 98.000 | +1.000 |
