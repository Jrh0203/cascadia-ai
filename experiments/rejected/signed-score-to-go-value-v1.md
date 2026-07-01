# Signed Score-To-Go Value V1

## Hypothesis

Predicting signed decomposed `final - current` score, then adding exact current
score back, would remove the burden of reconstructing the board's present
score and raise held-out final-score correlation above 0.50.

## Frozen Data

- Train: 256 H6 games, 20,480 positions, train indices 0-255.
- Validation: 64 H6 games, 5,120 positions, validation indices 0-63.
- Teacher: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`.
- Storage: 320 ordered, checksummed, one-game `.stg` shards.
- Identity: `current + residual = final` verified on every record.

Validation final totals span 82-99. Mean residual total declines from 86.70 at
turn 0 to 5.06 at turn 76. Signed values are required for Nature Tokens:
validation contains residuals as low as -5 after token spending.

## Best Checkpoint

Selected epoch 13 by reconstructed-final total MAE:

| Metric | Final target v1 | Score-to-go v1 |
|---|---:|---:|
| Total MAE | 2.538216 | 2.568601 |
| Total correlation | 0.211986 | 0.397451 |
| Total RMSE | 3.238570 | 3.266161 |
| Residual correlation | n/a | 0.991700 |

Wildlife MAEs were Bear 3.918660, Elk 3.054139, Salmon 3.394859, Hawk
3.140786, and Fox 3.000673. All stayed below the frozen ceilings.

The highest reconstructed-final correlation at any epoch was 0.414201 at
epoch 18, still below the 0.50 gate.

## Conclusion

Rejected before gameplay. Score-to-go learning accurately models phase and
remaining workload, but reconstructed final score still depends on a narrow
game-outcome residual. The semantic change nearly doubled correlation, yet did
not qualify the model as a leaf evaluator. No checkpoint was promoted.
