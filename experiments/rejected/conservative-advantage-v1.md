# Conservative Advantage V1: Rejected Before Test

The paired c90 lower-bound regression experiment completed its registered
collection and training protocol:

- 128 train games, 2,560 groups, 38,381 challengers;
- 32 validation games, 640 groups, 10,051 challengers;
- 32 checksummed untouched test games, 640 groups, 9,813 challengers;
- 17 MLX epochs and 4,352 optimizer steps before validation patience.

Epoch 12 was the selected checkpoint:

| Metric | Best validation | Gate | Result |
|---|---:|---:|---|
| MSE | 0.966878 | < 4.789431 zero predictor | pass |
| Mean policy regret | 0.095716 | <= 0.20 | pass |
| Exact policy agreement | 0.765625 | >= 0.65 | pass |
| Anchor false-positive rate | 0.004073 | <= 0.20 | pass |
| Selected-challenger recall | 0.006711 | >= 0.35 | fail |
| Lower-bound correlation | 0.760583 | >= 0.50 | pass |

The model learned broad lower-bound structure but collapsed to the
pattern-aware anchor. This was not just a threshold problem: with the
threshold removed, the selected challenger was still the top-ranked
challenger in only 16.8% of challenger-selected validation groups.

The recall gate failed before test access. The untouched test labels remain
unevaluated, no gameplay benchmark ran, and no model was promoted.

The successor changes the objective to balanced groupwise policy learning
over the anchor and all challengers while retaining lower-bound regression as
an auxiliary task.
