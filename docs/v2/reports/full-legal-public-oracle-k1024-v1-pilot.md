# Full-Legal Public Oracle K1024 V1 Pilot

- Status: **pilot_failed**
- Games: 12
- Baseline mean: 95.562 [94.812, 96.250]
- Treatment mean: 98.417 [97.104, 99.625]
- Paired delta: **2.854 [1.583, 4.188]**
- Record: 10 wins, 0 ties, 2 losses
- Mean local champion regret: 0.313
- Action change rate: 64.062%
- Top-1024 winner rate: 99.375%

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
| john1 | 4 | 94.812 | 98.188 | +3.375 |
| john2 | 4 | 96.125 | 98.062 | +1.938 |
| john3 | 4 | 95.750 | 99.000 | +3.250 |

## Runtime And Utilization

- Baseline mean decision latency: 194.674 ms
- Treatment mean decision latency: 4578.745 ms
- Aggregate cluster throughput: 27.152 games/hour

| Host | Wall s | Productive s | Games/hour | Max RSS MiB | Process swaps | System swap delta MiB | Retries |
|---|---:|---:|---:|---:|---:|---:|---:|
| john1 | 1485.010 | 1484.933 | 9.697 | 289.4 | 0 | -224.1 | 1 |
| john2 | 1507.300 | 1507.226 | 9.554 | 283.0 | 0 | 0.0 | 1 |
| john3 | 1591.060 | 1590.990 | 9.051 | 328.0 | 0 | 0.0 | 1 |

## Phase Diagnostics

| Phase | Decisions | Mean regret | Action change |
|---|---:|---:|---:|
| Early | 336 | 0.490 | 75.000% |
| Middle | 336 | 0.313 | 66.667% |
| Late | 288 | 0.107 | 48.264% |

## Seed Pairs

| Seed | Host | Baseline | Treatment | Delta |
|---:|---|---:|---:|---:|
| 62040 | john1 | 92.750 | 100.500 | +7.750 |
| 62041 | john1 | 94.750 | 98.500 | +3.750 |
| 62042 | john1 | 94.250 | 93.750 | -0.500 |
| 62043 | john1 | 97.500 | 100.000 | +2.500 |
| 62044 | john2 | 96.250 | 101.500 | +5.250 |
| 62045 | john2 | 96.000 | 95.000 | -1.000 |
| 62046 | john2 | 96.750 | 99.250 | +2.500 |
| 62047 | john2 | 95.500 | 96.500 | +1.000 |
| 62048 | john3 | 96.500 | 100.500 | +4.000 |
| 62049 | john3 | 96.500 | 98.750 | +2.250 |
| 62050 | john3 | 94.250 | 97.500 | +3.250 |
| 62051 | john3 | 95.750 | 99.250 | +3.500 |
