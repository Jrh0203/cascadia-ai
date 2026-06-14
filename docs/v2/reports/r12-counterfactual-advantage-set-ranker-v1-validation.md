# ADR 0078 R12 Counterfactual-Advantage Set Ranker

Status: **rejected on validation**.

Checkpoint: `step-000000000-epoch-0000-batch-000000`

| Metric | Model | Frozen comparison |
|---|---:|---:|
| Validation objective | 0.703511 | 0.703511 |
| Centered MAE | 0.643168 | 0.643168 |
| Centered correlation | 0.749648 | 0.749648 |
| Top-value recall | 0.449219 | 0.482422 |
| Mean top-action regret | 0.492676 | 0.438965 |

## Gates

- PASS: `mlx_gpu_device`
- FAIL: `validation_objective_improves_at_least_10_percent`
- PASS: `centered_mae_at_most_0_75`
- FAIL: `centered_mae_improves_at_least_10_percent`
- PASS: `centered_correlation_at_least_0_55`
- FAIL: `top_value_recall_at_least_0_50`
- FAIL: `top_value_recall_at_least_h6_plus_0_05`
- FAIL: `mean_regret_at_most_0_40`
- FAIL: `mean_regret_at_least_0_05_below_h6`
- PASS: `selected_metrics_match_best_pointer`
- PASS: `source_matches_training_run`

Test and gameplay domains remain closed.
