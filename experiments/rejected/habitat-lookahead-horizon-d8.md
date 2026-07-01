# Habitat Lookahead Horizon D8

Status: rejected at the pilot gate on 2026-06-10.

## Pilot

Ten paired games used seeds 22400-22409 and changed only the H6 rollout
horizon:

- H6/R4/D4 mean: 91.625
- H6/R4/D8 mean: 91.575
- paired delta: -0.050
- 95% CI: [-1.051, +0.951]
- habitat delta: -0.275
- total wildlife delta: +0.125
- treatment runtime: 6.772 seconds per game

## Conclusion

Observing a second future turn for the acting seat did not improve the greedy
rollout signal and increased runtime by 68%. Deeper greedy trajectories are
closed for this search family. Future strength work needs a better learned
leaf or rollout policy rather than more of the same policy.
