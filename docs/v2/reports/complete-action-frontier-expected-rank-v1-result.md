# ADR 0100 Frontier Expected-Rank Result

Classification: `expected_rank_optimization_underfit`

| Metric | Screen baseline | Selected model | Gate |
|---|---:|---:|---:|
| validation expected-rank target recall | 25.99% | 27.81% | 50.00% |
| validation exact expected-rank sets | 0.00% | 0.42% | 1.00% |
| validation R4800 winner recall | 72.92% | 73.33% | >98.00% |
| validation confidence coverage | 88.75% | 88.33% | 99.00% |
| validation retained regret | 0.070648 | 0.070215 | <0.030000 |

Train fit:

- expected-rank target recall: 32.21%;
- exact expected-rank target sets: 0.18%.

The independent cache pair was byte-identical: True. The widest-group gradient audit passed: True. The selected checkpoint replay was bit-identical: True.

The campaign completed in 2781.77 seconds with one MLX trainer and zero duplicate training.

The sealed test, gameplay, new teacher compute, cloud, and external compute remained closed.

Exploratory mechanism audit:

- validation target probability mass inside the deployed set: 45.51%;
- validation uniform-start absolute gradient inside the deployed set: 26.12%;
- validation exact-set reachability at residual range 6: 100.00%.
- validation deployed-set target mass rises from 45.51% at scale 64 to 93.75% at scale 16.

These diagnostics were opened after launch to explain the result and do not alter the preregistered classification.

Failed gates:

- `train_expected_rank_target_recall_at_least_0_80`
- `train_expected_rank_exact_sets_at_least_0_25`
- `validation_expected_rank_target_recall_at_least_0_50`
- `validation_expected_rank_exact_sets_at_least_0_01`
- `validation_r4800_winner_recall_strictly_above_0_98`
- `validation_confidence_coverage_at_least_0_99`
- `validation_distinguishable_recall_at_least_0_98`
- `validation_retained_regret_below_0_03`
- `early_winner_recall_at_least_0_98`
- `early_confidence_coverage_at_least_0_98`
- `early_retained_regret_below_0_03`
- `late_winner_recall_at_least_0_98`
- `late_confidence_coverage_at_least_0_98`
- `late_retained_regret_below_0_03`
- `middle_winner_recall_at_least_0_98`
- `middle_confidence_coverage_at_least_0_98`
- `middle_retained_regret_below_0_03`
- `nature_token_available_winner_recall_at_least_0_95`
- `independent_draft_winner_winner_recall_at_least_0_95`
- `pilot_passed`
