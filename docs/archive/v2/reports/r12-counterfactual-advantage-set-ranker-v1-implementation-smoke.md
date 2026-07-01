# ADR 0078 R12 Counterfactual-Advantage Set Ranker

Status: **rejected on validation**.

Checkpoint: `step-000000001-epoch-0001-batch-000000`

| Metric | Model | Frozen comparison |
|---|---:|---:|
| Validation objective | 0.447365 | 0.593250 |
| Centered MAE | 0.554363 | 0.554688 |
| Centered correlation | 0.902941 | 0.902894 |
| Top-value recall | 0.250000 | 0.250000 |
| Mean top-action regret | 0.208332 | 0.354166 |

## Gates

- PASS: `mlx_gpu_device`
- PASS: `validation_objective_improves_at_least_10_percent`
- PASS: `centered_mae_at_most_0_75`
- FAIL: `centered_mae_improves_at_least_10_percent`
- PASS: `centered_correlation_at_least_0_55`
- FAIL: `top_value_recall_at_least_0_50`
- FAIL: `top_value_recall_at_least_h6_plus_0_05`
- PASS: `mean_regret_at_most_0_40`
- PASS: `mean_regret_at_least_0_05_below_h6`
- PASS: `selected_metrics_match_best_pointer`
- PASS: `source_matches_training_run`

Test and gameplay domains remain closed.
