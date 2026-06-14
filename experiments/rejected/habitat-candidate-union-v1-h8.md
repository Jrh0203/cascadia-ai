# Habitat Candidate Union V1 H8

Status: rejected on runtime at the pilot gate on 2026-06-10.

## Design

The treatment unioned exact-immediate K8 with eight distinct draft-and-tile
placements ranked by:

1. exact matching terrain edges on the resulting board;
2. current total largest-habitat score;
3. exact immediate base score.

The metric changed candidate recall only. The unchanged fair R4/D4 rollout
evaluator selected the move.

## Pilot

Ten paired games against promoted K8 used seeds 21900-21909:

- K8 mean: 90.150
- K8+H8 mean: 91.725
- paired delta: +1.575
- 95% CI: [+0.516, +2.634]
- game record: 8 wins, 0 ties, 2 losses
- habitat delta: +0.925
- total wildlife delta: +0.425
- treatment runtime: 7.630 seconds per game

## Conclusion

Habitat-cohesion recall is the first candidate intervention to improve both its
target component and total score without a wildlife collapse. H8 nevertheless
missed the preregistered seven-second runtime gate, so it is rejected as
configured and received no confirmation.

A separately registered H4 cost-control variant tests whether the four
strongest cohesion placements retain the signal within budget.
