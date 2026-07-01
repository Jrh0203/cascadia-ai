# Habitat Candidate Wide Frontier V1: K16+H8

## Hypothesis

The exact union of generic immediate-score K16 and habitat-cohesion H8 might
combine both useful frontiers without the ranking error observed in the MLX
prefilter experiment.

## Pilot

Ten paired games compared frozen H6 against K16+H8 with identical R4/D4
evaluation:

- Baseline mean: 92.025
- Treatment mean: 91.700
- Paired delta: -0.325
- 95% confidence interval: -1.652 to 1.002
- Habitat delta: +0.325
- Wildlife delta: -0.450
- Nature Token delta: -0.200
- Record: 3-0-7
- Treatment runtime: 10.35 seconds per game

## Conclusion

Rejected at the score gate. The wider exact frontier was computationally
acceptable but did not improve action selection under the frozen rollout
objective. No confirmation run was performed.
