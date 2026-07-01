# ADR 0012: Test a Terminal Policy-Improvement Teacher

Status: rejected on 2026-06-10.

## Context

The existing MLX ranker already observes all four boards in relative-seat
order, the complete public market, phase, Nature Tokens, wildlife counts, and
habitat sizes. Its H6 labels, however, stop after four future plies and use the
acting seat's exact score at that shallow leaf.

That target cannot directly distinguish setup that survives the rest of the
game from setup that merely looks promising for one round. The rejected
phase-capped commitment experiment reinforces the point: cross-turn Bear
signal exists, but optimistic local opportunity can buy Bear by surrendering
other wildlife and habitat.

## Decision

Build an expensive research teacher for one-step policy improvement:

1. Generate the frozen pattern-aware K8+H6+B8 root frontier.
2. Draw two public-information hidden-state redeterminizations.
3. Share those samples across every root candidate.
4. Apply one candidate to each sample.
5. Run frozen pattern-aware play for every seat until game end.
6. Rank candidates by mean terminal base score for the acting seat.

The actual hidden stack is never scored. Common random numbers reduce paired
candidate variance, and terminal labels include every later market interaction
and allocation consequence. The policy is expected to be too slow for product
use; its purpose is to test whether these labels are worth distilling into MLX.

## Qualification Protocol

The seed-24699 smoke must finish the R2 treatment within 600 seconds. Only then
may seeds 24700 through 24702 run. The three-game teacher qualification
requires:

- paired mean delta at least +1.0;
- total wildlife delta at least 0.0;
- habitat delta at least -0.5;
- Bear delta at least 0.0;
- treatment runtime at most 600 seconds per game.

Passing qualifies a separately registered terminal-label collection and
training experiment. It does not promote the online teacher to the product.

## Outcome

The smoke passed at 100.334 treatment seconds. The three-game qualification
then scored 92.167 against 91.917:

- paired delta: +0.250, 95% CI `[-4.458, 4.958]`;
- per-seed deltas: -2.75, -1.50, and +5.00;
- record: 1-0-2;
- Bear: +1.250;
- total wildlife: +1.167;
- habitat: +0.917;
- Nature Tokens: -1.833;
- treatment runtime: 50.545 seconds per game.

The category mechanism was constructive rather than reallocative, but the
primary +1.0 qualification gate failed and variance was extreme. R2 is not a
reliable terminal-label teacher, so no dataset collection was permitted.
