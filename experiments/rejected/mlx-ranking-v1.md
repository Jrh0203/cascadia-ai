# MLX Ranking V1

Status: rejected as a standalone policy on 2026-06-10.

## Training

The frozen K8+B8/R4/D4 teacher labeled:

- 128 train games, 10,240 groups, and 116,310 candidates;
- 32 validation games, 2,560 groups, and 29,137 candidates.

`entity-set-ranker-v1` used hidden size 96, four attention heads, two board
blocks, one market block, listwise cross-entropy, AdamW, and learning rate
`1e-4`. Training stopped at epoch 10 after five consecutive non-improving
validation epochs. Epoch 5 was promoted.

Held-out metrics:

- listwise loss: 2.263
- top-1 accuracy: 0.232
- mean top-1 teacher regret: 0.250
- pairwise accuracy: 0.800
- mean rank correlation: 0.508
- value-difference correlation: 0.663

All preregistered ranking-quality gates passed.

## Gameplay

The one-game smoke scored 90.0 at 0.60 seconds per game.

Twenty paired games against K8 used seeds 21600-21619:

- K8 mean: 90.863
- MLX ranking mean: 88.638
- paired delta: -2.225
- 95% CI: [-2.986, -1.464]
- game record: 3 wins, 0 ties, 17 losses
- treatment runtime: 0.525 seconds per game

The student gained 2.225 Bear points but lost 1.41 Elk, 1.41 Salmon, 0.71
Hawk, and 1.28 Fox points.

## Conclusion

Good local ranking metrics did not imply complete-game policy parity. Small
per-decision regret compounded and the model reproduced the teacher's targeted
Bear preference without the rollout evaluator's balancing effect.

The standalone policy is rejected. The promoted artifact remains eligible only
for registered experiments as a candidate prefilter before fair rollout
evaluation.
