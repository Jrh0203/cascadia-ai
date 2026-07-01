# MLX Ranking Prefilter V1

Status: rejected at the pilot gate on 2026-06-10.

## Design

The promoted `entity-ranker-v1-k8b8` model ranked the full K8+B8 candidate
union. The top eight model actions then went through the unchanged R4/D4 fair
rollout evaluator. The model never supplied a rollout leaf value and never saw
hidden simulator state.

## Pilot

Ten paired games against promoted K8 used seeds 21700-21709:

- K8 mean: 90.275
- MLX-prefiltered mean: 92.000
- paired delta: +1.725
- 95% CI: [+0.528, +2.922]
- game record: 7 wins, 2 ties, 1 loss
- treatment runtime: 4.975 seconds per game

The treatment gained 2.70 Bear, 0.75 total habitat, and 0.825 remaining Nature
Tokens. It lost 2.55 combined Elk, Salmon, Hawk, and Fox points.

## Conclusion

The total-score signal was strong, but the experiment failed its preregistered
non-Bear gate. The ranker can remove too much of the balanced immediate-score
frontier when it owns all eight prefilter slots.

No confirmation was run. The next registered variant protects six exact
immediate-score anchors and gives MLX only two replacement slots.
