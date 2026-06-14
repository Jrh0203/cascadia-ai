# H6 Apprentice Policy Iteration V1

Status: rejected before gameplay on 2026-06-10.

## Hypothesis

The first H6 ranker was trained only on H6-visited states. Collecting H6
counterfactual labels on states visited by the ranker's own standalone policy,
then warm-starting on the aggregate distribution, should reduce compounding
errors without forgetting the original teacher distribution.

## Data

- Original H6 train: 128 games, 10,240 groups.
- New apprentice-trajectory train: 64 games, 5,120 groups, 64,795 candidates.
- New apprentice-trajectory validation: 16 games, 1,280 groups, 16,309
  candidates.
- Original H6 validation regression set: 32 games, 2,560 groups.

Every new manifest binds the trajectory to apprentice model manifest
`6fa654a6af96fa8cf137a5b57d808c311565bbf2d61a8ebee50c2f511218c723`.

## Training

The 96-dimensional `entity-set-ranker-v1` was warm-started with a fresh AdamW
optimizer at learning rate `3e-5`, group batch 16, and patience 3. Checkpoint
selection averaged listwise loss over apprentice and original validation.
Training stopped after seven epochs and 6,720 steps.

## Best Result

| Metric | Iteration 0 | Best epoch 4 | Change |
|---|---:|---:|---:|
| Balanced selection loss | 2.428105 | 2.425570 | -0.002535 |
| Apprentice top-one regret | 0.370703 | 0.352930 | -0.017773 |
| Apprentice pairwise accuracy | 0.777293 | 0.779077 | +0.001784 |
| Original H6 top-one regret | 0.334277 | 0.327734 | -0.006543 |
| Original H6 pairwise accuracy | 0.792058 | 0.792670 | +0.000611 |

## Conclusion

The iteration improved both distributions and did not forget, but the
apprentice regret gain missed the preregistered `0.03` gate. No model was
promoted and no gameplay pilot was run.

Distribution shift is real but not the dominant bottleneck for this model and
target. The next iteration should strengthen the target or representation,
especially cross-turn wildlife commitments and opponent/market interaction,
rather than collecting another identical DAgger round.

Artifacts:

- `artifacts/datasets/ranking-h6-iteration1-train/dataset.json`
- `artifacts/datasets/ranking-h6-iteration1-validation/dataset.json`
- `artifacts/runs/entity-ranker-v1-h6-iteration1/initial-validation.json`
- `artifacts/runs/entity-ranker-v1-h6-iteration1/best.json`
- `artifacts/runs/entity-ranker-v1-h6-iteration1/final-report.json`
