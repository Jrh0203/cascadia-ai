# Rank-Stratified Counterfactual Contrast Audit

- Dataset: `counterfactual-advantage-habitat-candidate-lookahead-v1-k8-h6-r4-d4-selected-high-median-low-v1-k4-g16-r16-validation-67000`
- Games / groups / candidates / continuations: 2 / 32 / 128 / 2048
- R8 centered MAE to R16: 0.3525
- R8 centered correlation: 0.9305
- R8 pairwise accuracy: 85.42%
- R8 exact top-action agreement: 62.50%
- R8 mean top-action regret: 0.1445
- Mean R16 group range: 2.8027
- Mean R16 centered-advantage SE: 0.4462
- Projected 160-game R8 corpus: 7.10 hours
- Failed gates: r8_top_agreement_at_least_0_65
- Verdict: **FAIL**
