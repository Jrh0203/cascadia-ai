# ADR 0010: Test Two-Turn Wildlife Commitment

Status: experiment preregistered on 2026-06-10.

## Context

The promoted pattern policy is close to the independently reproduced v1
reference across Elk, Salmon, Hawk, Fox, and Nature Tokens, but trails by about
4.0 Bear points and 2.1 habitat points. Its opportunity term sees only the
best exact marginal from one future wildlife market.

Bear A exposes the horizon error directly. After one pair is complete, the
first Bear of pair two scores no Bear points; the following Bear adds seven.
A one-turn marginal cannot value that bridge.

## Decision

Keep the exact same unified K8+H6+B8 frontier and replace only the opportunity
value with a two-personal-turn Bellman recursion:

1. For each wildlife species, enumerate every legal exact placement.
2. Add its exact immediate base-score gain.
3. Decrement that public species supply by one.
4. Add the exact one-turn expected best opportunity from the resulting board.
5. Compute the expected maximum species value from four draws without
   replacement.

The model is public-information-only and contains no species weights. It
assumes unchosen market wildlife remains in public supply, so it is an
optimistic opportunity model rather than a hidden-state simulation.

The full strategy must first complete one paired game within five seconds.
The ten-game pilot then requires +0.5 total, +0.5 Bear, no more than -0.5
aggregate non-Bear wildlife, no more than -0.5 habitat, and the same runtime
ceiling. Only a passing frozen pilot may advance to 50 disjoint games.

## V1 Outcome

The runtime smoke passed, but the ten-game pilot scored -0.675 paired with
only +0.100 Bear and -0.825 habitat. The implementation also valued two future
placements when the acting seat had only one turn left. V1 is rejected as
measured. A phase-capped correction must use a new strategy and experiment ID.
