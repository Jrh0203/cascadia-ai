# Pattern-Aware V1

## Hypothesis

A one-turn opportunity value in exact score units could reward wildlife setup
without category weights. Each retained action was valued by its exact
post-action base score plus the expected best legal one-token marginal from a
four-token market drawn without replacement from public unplaced supply.

## Pilot

Ten paired games against exact greedy produced:

- Baseline mean: 88.600
- Treatment mean: 92.175
- Paired delta: +3.575
- 95% confidence interval: 2.909 to 4.241
- Habitat delta: +0.925
- Wildlife delta: +1.375
- Nature Token delta: +1.275
- Bear delta: +4.125
- Record: 10-0-0
- Treatment runtime: 2.291 seconds per game

## Conclusion

Rejected as configured because it missed the preregistered two-second runtime
gate. The strength result is compelling enough to justify a separately
registered, behavior-preserving cost-control implementation after profiling.
The gate is not relaxed and no confirmation was run.
