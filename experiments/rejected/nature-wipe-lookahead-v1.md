# Nature-Wipe Lookahead V1

Status: rejected on 2026-06-10.

## Hypothesis

The promoted K8 policy leaves useful Nature Token value unrealized because it
never considers paying to refresh visible wildlife. A fair chance node could
select refreshes by expected value before observing replacement draws.

## Treatment

Before each placement, the strategy:

1. applies any mandatory free three-of-a-kind replacement;
2. compares no paid wipe with all 15 legal one-wipe slot subsets;
3. redetermines the hidden wildlife bag twice for every visible option;
4. evaluates the best of four immediate actions followed by four greedy plies;
5. commits to the highest expected-value option before observing the actual
   replacement; and
6. runs the promoted K8 placement search on the resulting public market.

The leaf score included the remaining Nature Tokens, so every wipe carried its
exact one-point terminal cost. Repeated paid wipes within one turn were not
considered.

## Protocol And Result

Five paired AAAAA four-player games used seeds 21100-21104:

```bash
target/release/cascadia-v2 nature-wipe-compare \
  --games 5 --first-seed 21100 \
  --candidates 8 --determinizations 4 --greedy-plies 4 \
  --prelude-candidates 4 --prelude-determinizations 2 \
  --prelude-greedy-plies 4 \
  --output docs/v2/reports/nature-wipe-lookahead-v1-pilot5.json
```

- baseline mean: 89.850
- treatment mean: 89.350
- paired delta: -0.500
- 95% CI: [-2.410, 1.410]
- game record: 3 wins, 0 ties, 2 losses
- paid wipes: 60 across 20 seat-games
- wildlife slots replaced: 142
- runtime: 66.44 seconds

Bear improved by 2.25 points, but Nature Tokens fell by 1.30 and Elk, Salmon,
Hawk, and Fox all regressed.

## Conclusion

The information ordering is correct, but this value approximation is not.
Two hidden-state samples and a four-ply leaf make marginal refresh differences
easy to overestimate, causing twelve paid wipes per game on average. The
experiment is rejected and is not part of the product strategy.

Future market-control work needs either calibrated search-teacher targets or a
strict exercise threshold learned on held-out outcomes. Increasing compute on
this uncalibrated objective is not justified.
