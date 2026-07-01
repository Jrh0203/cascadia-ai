# ADR 0013: Increase Terminal Policy Samples

Status: teacher qualified on 2026-06-10.

## Context

R2 terminal policy improvement failed its +1.0 qualification gate at +0.250.
Its category changes were nevertheless unlike the earlier allocation failures:
Bear improved 1.250, total wildlife improved 1.167, and habitat improved 0.917.
The three paired game deltas were -2.75, -1.50, and +5.00.

The operator is a standard one-step policy improvement estimate. With exact
expected continuation values it should not weaken its frozen continuation
policy, but with only two stochastic hidden-state samples, root candidates can
be selected on sampling error.

## Decision

Run one variance-control test with eight shared determinizations. Keep every
other boundary frozen:

- pattern-aware K8+H6+B8 root frontier;
- public-information redeterminizations;
- common random numbers across root candidates;
- pattern-aware continuation for every seat;
- acting-seat terminal base-score target.

The seed-24899 smoke must finish within 600 treatment seconds. Only then may
seeds 24900 through 24902 run. Qualification requires +1.0 paired score,
nonnegative total wildlife, at least -0.5 habitat, nonnegative Bear, at least
-1.0 Nature Tokens, and the same runtime ceiling.

Passing qualifies a separately registered terminal-label collection. Failure
closes terminal policy improvement at this local full-game sampling regime.

## Outcome

The smoke passed at 273.884 treatment seconds and +2.250 paired. The registered
three-game qualification then scored 94.833 against 93.500:

- paired delta: +1.333, 95% CI `[-2.249, 4.916]`;
- record: 2-0-1;
- Bear: +1.750;
- total wildlife: +0.333;
- habitat: +1.417;
- Nature Tokens: -0.417;
- treatment runtime: 185.878 seconds per game.

Every registered teacher gate passed. The small sample and wide interval do
not justify product promotion, but R8 is qualified to generate long-horizon
ranking labels for a separately registered MLX distillation experiment.
