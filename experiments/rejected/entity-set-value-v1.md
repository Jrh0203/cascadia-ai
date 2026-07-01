# Entity-Set Value V1

Status: rejected on 2026-06-10.

## Hypothesis

A decomposed entity-set attention model trained on fresh canonical greedy
trajectories can rank every legal afterstate well enough to produce a
competent pure learned policy.

## Training

- 256 training games and 20,480 positions
- 64 disjoint validation games and 5,120 positions
- MLX 0.31.2 on the local Apple GPU
- 20 epochs, batch size 256, AdamW at 0.0003
- best checkpoint: epoch 8
- held-out total MAE: 2.817
- held-out total RMSE: 3.491
- held-out total correlation: 0.163

The model and optimizer checkpoints are reproducible under
`artifacts/runs/entity-value-v1-greedy256`. The validated standalone artifact
is `artifacts/models/entity-value-v1-greedy256`.

## Gameplay Gate

The Rust/MLX boundary was corrected before the decisive run so every candidate
record represented a complete legal transition: updated board, market, token
count, phase, and acting-seat perspective.

Command:

```bash
target/release/cascadia-v2 model-benchmark \
  --games 1 \
  --first-seed 10000 \
  --model-dir artifacts/models/entity-value-v1-greedy256 \
  --output docs/archive/v2/reports/mlx-value-v1-true-afterstate-smoke-1.json
```

Result:

- mean: 41.25
- seat range: 33-48
- runtime: 97.62 seconds

Every seat failed the predeclared 75-point competence floor, so the planned
10-game trial was stopped for futility.

## Conclusion

Low absolute error did not imply action-ranking quality. Greedy trajectories
produce a narrow final-score distribution, allowing the regressor to predict
the mean while remaining poorly correlated with position quality. Maximizing
that regressor over thousands of off-policy candidates exposed severe
extrapolation error.

Future learned action selection must report ranking metrics and control
distribution shift through candidate filtering, counterfactual action labels,
search targets, or policy objectives. More epochs on this objective are not a
valid follow-up.
