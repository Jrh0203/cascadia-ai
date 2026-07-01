# Conservative Policy V2: Rejected Before Test

The balanced groupwise successor ran on the exact frozen v1 train and
validation manifests. It stopped after nine epochs and 2,304 optimizer steps
at validation patience. Epoch 4 was selected.

| Metric | Best validation | Gate | Result |
|---|---:|---:|---|
| Balanced policy cross-entropy | 1.892260 | selection metric | best |
| MSE | 1.498838 | < 4.789431 zero predictor | pass |
| Mean policy regret | 0.089396 | <= 0.20 | pass |
| Exact policy agreement | 0.767188 | >= 0.65 | pass |
| Anchor false-positive rate | 0.000000 | <= 0.20 | pass |
| Selected-challenger recall | 0.000000 | >= 0.35 | fail |
| Lower-bound correlation | 0.669211 | >= 0.50 | pass |

The policy head retained the anchor in every validation group. Its selected
challenger was the highest-logit challenger in only 15.4% of
challenger-selected groups, so threshold calibration could not repair it.

The failure exposed a target mismatch: terminal sample seeds are derived from
the hidden game seed, but the model input intentionally stops at the public
pre-refill boundary. Exact R8 choices contain sample-specific Monte Carlo
noise unavailable to the learner.

The untouched test labels remain unevaluated. No gameplay ran and no model was
promoted. Further loss engineering on the same R8 labels is closed; the next
experiment measures whether an R32 conservative teacher is stronger and
stable enough to justify recollection.
