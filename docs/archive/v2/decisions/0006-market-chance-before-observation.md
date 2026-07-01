# ADR 0006: Market Chance Must Precede Observation

Status: accepted on 2026-06-10.

## Context

A paid wildlife refresh is a sequential decision:

1. spend a Nature Token and name the visible slots to replace;
2. draw hidden wildlife;
3. observe the new public market; and
4. choose a draft and placement.

Generating a complete atomic `TurnAction` directly from the simulator's true
hidden bag would let the policy choose whether to pay after seeing the result.
That would be legal to execute but invalid as research evidence.

## Decision

Search evaluates refresh choices by redetermining the unseen bag before each
replacement. It averages each visible wipe option across common hidden-state
samples and commits to the option on expected value alone.

Only after that option is fixed does the strategy preview the actual
replacement, which is now public, and run the normal placement search. The
returned atomic action records both decisions so replay and rules execution
remain canonical.

The first experiment considers no wipe and every legal single paid wipe. It
does not assume that repeated wipes are useful; multi-wipe planning requires
separate evidence.

## Consequences

The policy respects the same information sequence as a human player and can be
reproduced through seeded replay. Prelude search is more expensive than normal
placement search, so it remains experimental until paired strength and runtime
gates are met.

The first implementation was rejected in its five-game pilot: it selected 60
paid wipes across 20 seat-games and lost 0.50 points despite a 2.25-point Bear
gain. This does not invalidate the information-ordering rule; it shows that a
short-horizon, two-sample leaf estimate is not calibrated well enough to price
the option to refresh. The strategy remains available only as reproducible
research code.
