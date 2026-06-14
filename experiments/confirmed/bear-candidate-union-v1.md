# Bear Candidate Union V1

Status: confirmed as a research teacher on 2026-06-10; not promoted to the
interactive product policy.

## Hypothesis

Immediate-score top-K filtering excludes actions that create exact A-card Bear
pair completion slots. Adding Bear-ready candidates without changing the
rollout value should recover that loss without repeating the rejected
handcrafted-potential experiment.

## Treatment

`bear-candidate-lookahead-v1-k8-b8-r4-d4` unions:

- the eight highest exact immediate-score actions; and
- up to eight actions ranked by resulting Bear score, exact pair-ready slots,
  and immediate total.

The union is deduplicated. Every retained action is still selected by the same
four-determinization, four-greedy-ply public-information rollout objective as
the promoted K8 policy. Bear readiness changes recall only; it is never added
to the leaf value.

## Results

Against product K8 on disjoint seeds 21300-21349:

- baseline: 90.350
- treatment: 91.215
- paired delta: +0.865
- 95% CI: [+0.320, +1.410]
- game record: 31 wins, 5 ties, 14 losses
- Bear delta: +3.560
- treatment runtime: 5.544 seconds per game

Against generic K16 on disjoint seeds 21400-21449:

- baseline: 91.610
- treatment: 91.730
- paired delta: +0.120
- 95% CI: [-0.346, +0.586]
- game record: 24 wins, 3 ties, 23 losses
- Bear delta: +2.560
- treatment runtime: 4.144 seconds per game

## Conclusion

Species-aware candidate generation is a real improvement over K8 and directly
repairs much of the Bear deficit. It does not outperform generic K16 at equal
maximum breadth, so the pre-registered product-promotion control failed.

The strategy is retained as a confirmed research teacher because it is
competitive with K16, faster on the control suite, and provides the targeted
candidate diversity needed for search distillation. The next search experiment
tests the actual K16+B8 superset rather than replacing generic candidates.
