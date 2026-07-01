# Same-Decision Counterfactual Advantage Target Audit

- Dataset: `counterfactual-advantage-habitat-candidate-lookahead-v1-k8-h6-r4-d4-k4-g16-r16-validation-66000`
- Games / groups / candidates / continuations: 2 / 32 / 128 / 2048
- R8 centered MAE to R16: 0.2739
- R8 centered correlation: 0.8549
- R8 pairwise accuracy: 89.58%
- R8 exact top-action agreement: 81.25%
- R8 mean top-action regret: 0.0566
- Mean R16 group range: 1.3672
- Mean R16 centered-advantage SE: 0.3843
- Projected 160-game R8 corpus: 7.72 hours
- Failed gates: mean_group_range_at_least_1_50
- Verdict: **FAIL**

R8 passed every stability, ranking, regret, uncertainty, integrity, and local
runtime gate. The selected action and its three nearest H6 alternatives were
too similar, however: their R16 means spanned only 1.3672 points on average
against the frozen 1.50-point minimum.

The target is rejected without rounding. No train corpus, model, sealed test,
or gameplay comparison is authorized. A successor must preregister broader
rank-stratified contrasts from the existing H6 frontier rather than resampling
this narrow top-four set.
