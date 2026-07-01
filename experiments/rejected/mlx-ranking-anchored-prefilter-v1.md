# MLX Ranking Anchored Prefilter V1

Status: rejected at the pilot gate on 2026-06-10.

## Design

The treatment always retained exact-immediate K8 ranks one through six. The
promoted MLX ranker filled only the last two slots from the remaining K8+B8
union, after which the unchanged R4/D4 rollout evaluator selected the move.

## Pilot

Ten paired games against promoted K8 used seeds 21800-21809:

- K8 mean: 90.550
- anchored-prefilter mean: 91.425
- paired delta: +0.875
- 95% CI: [-0.393, +2.143]
- game record: 6 wins, 1 tie, 3 losses
- treatment runtime: 3.996 seconds per game

The treatment gained 3.05 Bear, 0.40 habitat, and 0.40 remaining Nature Tokens.
It lost 2.975 combined Elk, Salmon, Hawk, and Fox points.

## Conclusion

Protecting six exact-immediate actions constrained candidate displacement but
did not correct the learned species tradeoff. Even two model-owned slots
materially shifted play away from the four non-Bear cards.

The variant failed its preregistered mechanism gate, so no confirmation was
run. Further ranker prefilter tuning is stopped pending a materially different
training objective or teacher.
