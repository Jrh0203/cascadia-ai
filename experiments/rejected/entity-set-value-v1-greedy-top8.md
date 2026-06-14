# Entity-Set Value V1 With Greedy Top-8

Status: rejected on 2026-06-10.

## Hypothesis

The rejected final-score regressor may still contain useful local action
ranking signal if exact immediate-score greedy restricts inference to eight
near-frontier candidates.

## Protocol

Twenty paired deterministic games used seeds 10000-10019. Both strategies
played all four seats under AAAAA scoring with habitat bonuses disabled.

Treatment:

1. enumerate every canonical legal action,
2. retain the eight highest immediate base-score afterstates,
3. construct a complete post-transition record for each,
4. choose the largest MLX-predicted final score.

Command:

```bash
target/release/cascadia-v2 model-compare \
  --games 20 \
  --first-seed 10000 \
  --baseline greedy \
  --model-dir artifacts/models/entity-value-v1-greedy256 \
  --prefilter-k 8 \
  --output docs/v2/reports/mlx-value-v1-greedy-top8-vs-greedy-20.json
```

## Result

- greedy mean: 87.125
- treatment mean: 84.438
- paired delta: -2.688
- 95% CI: [-3.518, -1.857]
- game record: 2 wins, 0 ties, 18 losses
- elapsed: 138.52 seconds

The treatment lost 2.59 habitat points and 1.36 wildlife points on average,
partly offset by 1.26 more unused Nature Tokens.

## Conclusion

Candidate filtering prevents catastrophic extrapolation but does not turn this
objective into a useful policy. The next learned experiment needs
counterfactual action-quality or search targets, explicit action-ranking
metrics, and a held-out paired gameplay gate. Reusing this model as a policy
head is rejected.
