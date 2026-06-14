# MLX H6 Self-Rollout V1

## Hypothesis

The qualified H6 ranker might extend the acting player's own plan across one
full table rotation at an acceptable cost if the three opponent plies retained
the exact greedy policy.

## Runtime Gate

The preregistration required one complete K8/H6/R4/D4 treatment game at no
more than 30 seconds before any ten-game strength pilot.

The full configuration completed correctly with:

- Treatment wall time: 62.582 seconds
- Baseline wall time: 8.651 seconds
- Mean decision latency: 782 ms
- P90 decision latency: 1,567 ms
- Maximum decision latency: 5,525 ms
- Single-game paired delta: -2.5

## Conclusion

Rejected at the mandatory runtime gate. Limiting inference to one future
acting-seat turn was much cheaper than using MLX for every rollout seat, but
it still exceeded the registered ceiling by more than twofold. The ten-game
strength pilot was not run.
