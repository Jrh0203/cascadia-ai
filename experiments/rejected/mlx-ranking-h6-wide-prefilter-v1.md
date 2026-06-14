# MLX H6 Wide Prefilter V1

## Held-Out Qualification

The fresh H6 ranker selected epoch 5 after validation futility:

- Pairwise accuracy: 0.792
- Mean top-one regret: 0.334
- Value-difference correlation: 0.759

All preregistered model-quality gates passed.

## Gameplay Protocol

- Baseline: H6 K8+H6/R4/D4
- Treatment: eight exact-immediate anchors plus six MLX-selected additions
  from K16+H8, followed by unchanged R4/D4 evaluation
- Seeds: 22500-22509
- Games: 10 paired four-player AAAAA games

## Result

- Baseline mean: 91.625
- Treatment mean: 91.800
- Paired delta: +0.175
- 95% paired CI: -1.322 to 1.672
- Record: 3 wins, 0 ties, 7 losses
- Habitat delta: -0.325
- Wildlife delta: -0.075
- Nature Token delta: +0.575
- Treatment runtime: 7.337 seconds per game

## Conclusion

Rejected at the registered +0.25 gameplay gate. The anchored design avoided
the species collapse seen with the Bear-trained ranker, but it did not produce
enough total or habitat signal to justify a 50-game confirmation. The model
remains eligible for the separately registered rollout-policy experiment
because its held-out ranking gates passed.
