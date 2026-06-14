# Bear + Habitat Candidate Union V1

## Hypothesis

Adding Bear setup actions to H6's balanced habitat frontier would recover the
independently measured Bear deficit without discarding H6's habitat and
non-Bear gains.

## Protocol

- Baseline: `habitat-candidate-lookahead-v1-k8-h6-r4-d4`
- Treatment: `bear-habitat-candidate-lookahead-v1-k8-h6-b8-r4-d4`
- Seeds: 22800-22809
- Games: 10 paired four-player AAAAA games
- Shared evaluator: four public-information determinizations and four greedy
  future plies

## Result

- Baseline mean: 91.700
- Treatment mean: 91.400
- Paired delta: -0.300
- 95% paired CI: -2.048 to 1.448
- Record: 5 wins, 0 ties, 5 losses
- Bear delta: +2.075
- Aggregate Elk + Salmon + Hawk + Fox delta: -2.500
- Habitat delta: +0.075
- Treatment runtime: 8.146 seconds per game

## Conclusion

Rejected at the score and non-Bear mechanism gates. The wider union found Bear
points but paid for them almost exactly by abandoning other species. This is
the same allocation failure seen in the Bear-only teacher and ranker studies,
now reproduced against H6. Future Bear work must alter value or planning
quality rather than inject more Bear-ranked root actions.
