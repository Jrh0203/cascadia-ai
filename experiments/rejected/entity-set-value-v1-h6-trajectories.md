# Entity-Set Value V1 on H6 Trajectories

## Hypothesis

Replacing narrow greedy trajectories with confirmed H6 games would raise
held-out final-score correlation enough to support a cheap learned leaf value.

## Data

- Train: 256 H6 games, 20,480 positions
- Validation: 64 disjoint H6 games, 5,120 positions
- Targets: final acting-seat Bear, Elk, Salmon, Hawk, Fox, five habitat, and
  Nature Token components

## Best Checkpoint

- Epoch: 10
- Total MAE: 2.538
- Total RMSE: 3.239
- Total correlation: 0.212
- Calibration slope: 0.461

The model improved MAE over the greedy-trained baseline and did not regress any
wildlife component by more than the registered one-point allowance.

## Conclusion

Rejected before gameplay because total correlation missed the required 0.50
gate by a wide margin. Strong-policy outcomes are even more tightly
distributed, so low absolute error mostly reflects predicting the mean rather
than ranking future quality. The separately implemented value-leaf search was
not run with this unqualified checkpoint.
