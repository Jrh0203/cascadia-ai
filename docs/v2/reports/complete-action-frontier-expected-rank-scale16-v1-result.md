# ADR 0101 Frontier Expected-Rank Scale 16 Result

Classification: `scale16_alignment_insufficient`

| Metric | Screen baseline | Selected model | Gate |
|---|---:|---:|---:|
| validation expected-rank target recall | 25.99% | 27.21% | 50.00% |
| validation exact expected-rank sets | 0.00% | 0.00% | 1.00% |
| validation R4800 winner recall | 72.92% | 73.33% | >98.00% |
| validation confidence coverage | 88.75% | 89.58% | 99.00% |
| validation retained regret | 0.070648 | 0.071272 | <0.030000 |

Train fit:

- target recall: 30.23%;
- exact target sets: 0.18%;
- recall delta versus ADR 0100: -1.98%.

Alignment diagnostics:

- train deployed-set target mass: 93.76%;
- validation deployed-set target mass: 93.75%;
- validation absolute gradient inside deployed set: 47.86%.

The cache audit passed: True. The 12-group gradient audit passed: True. The reachability audit passed: True. The selected checkpoint replay was bit-identical: True.

The campaign completed in 2292.16 seconds with one trainer and zero duplicate training.

The sealed test, gameplay, new teacher compute, cloud, and external compute remained closed.

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
- `middle_winner_recall_at_least_0_98`
- `middle_confidence_coverage_at_least_0_98`
- `middle_retained_regret_below_0_03`
- `nature_token_available_winner_recall_at_least_0_95`
- `independent_draft_winner_winner_recall_at_least_0_95`
- `pilot_passed`
