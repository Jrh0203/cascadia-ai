# Bear Candidate Union V2: K16+B8

Status: rejected on 2026-06-10.

## Hypothesis

The actual K16+B8 candidate superset could preserve generic K16 breadth while
recovering exact Bear-ready actions that immediate-score ranking misses.

## Protocol And Result

Ten paired AAAAA four-player games used seeds 21500-21509. Both strategies
used four hidden-state determinizations and four greedy rollout plies.

- baseline K16 mean: 91.675
- treatment K16+B8 mean: 91.525
- paired delta: -0.150
- 95% CI: [-1.676, 1.376]
- game record: 5 wins, 0 ties, 5 losses
- Bear delta: +2.575
- combined Elk, Salmon, Hawk, and Fox delta: -2.000
- treatment runtime: 6.895 seconds per game

## Conclusion

More candidate breadth did not compound under the current rollout objective.
The treatment again traded other wildlife for Bear and failed every total-score
advancement gate. K8+B8 remains the lower-cost, confirmed research teacher for
distillation; K16+B8 is rejected.
