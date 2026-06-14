# ADR 0011: Phase-Cap Wildlife Commitment

Status: rejected on 2026-06-10.

## Context

The first two-turn wildlife-commitment experiment was rejected at -0.675
paired, +0.100 Bear, and -0.825 habitat. Its post-result mechanism audit also
found a semantic defect: the evaluator assigned two future personal wildlife
placements whenever any turn remained, including states where the acting seat
could make only one more placement before game end.

That impossible late-game setup value is large enough to change decisions, but
it does not erase or reinterpret the v1 result. The correction therefore needs
a new strategy identity, disjoint seeds, and a separately registered test.

## Decision

Keep the v1 frontier and opportunity model unchanged, except after applying
each candidate:

1. Read the acting seat's exact turns remaining from the transitioned state.
2. Evaluate `min(2, turns_remaining)` future personal placements.
3. Use zero opportunity when no turn remains.

At 72 completed plies in a four-player game, the current seat has two turns
including the current action. The corrected commitment ranking must therefore
be exactly equal to the one-turn pattern-aware ranking after that action. A
regression test asserts equality of the complete ranked candidate vectors.

## Experiment

The mandatory smoke uses seed 24399. If its treatment runtime is at most five
seconds, a ten-game pilot uses seeds 24400 through 24409 and requires:

- paired mean delta at least +0.5;
- Bear delta at least +0.5;
- aggregate non-Bear wildlife delta at least -0.5;
- habitat delta at least -0.5;
- treatment runtime at most five seconds per game.

Only a passing pilot may advance to the frozen 50-game confirmation on seeds
24500 through 24549.

## Outcome

The runtime smoke passed at 0.478 treatment seconds per game. On the registered
ten-game pilot, the corrected policy scored 92.225 against 91.575:

- paired delta: +0.650, 95% CI `[-0.167, 1.467]`;
- record: 8-0-2;
- Bear: +1.900;
- aggregate non-Bear wildlife: -0.950;
- total wildlife: +0.950;
- habitat: -0.650;
- Nature Tokens: +0.350;
- treatment runtime: 0.338 seconds per game.

Phase capping repaired the semantic defect and exposed a substantially stronger
Bear signal, but the treatment still reallocates too much value away from the
other wildlife cards and habitat. It failed both registered -0.5 mechanism
guardrails, so no confirmation was run.
