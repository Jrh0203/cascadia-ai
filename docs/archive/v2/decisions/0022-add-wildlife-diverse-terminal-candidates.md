# ADR 0022: Add Wildlife-Diverse Terminal Candidates

Status: rejected on 2026-06-11 after the registered pilot.

## Context

Final-five R8 search confirmed a +0.425 paired gain with a positive confidence
interval, but missed its frozen Bear guardrail by 0.050. Its root frontier
retains exact-immediate K8, habitat H6, and Bear B8 actions. It has no
equivalent coverage rule for Elk, Salmon, Hawk, or Fox.

Terminal evaluation cannot recover a strong action omitted before rollout.
The confirmed treatment's non-Bear wildlife delta was -0.055, consistent with
a frontier that explicitly protects Bear but not the other four species.

## Decision

Test one bounded candidate-information change:

1. Preserve every action in the frozen K8+H6+B8 frontier.
2. For each wildlife species visible after the market prelude, add up to two
   actions drafting that species.
3. Rank those additions by exact resulting score for that species, then exact
   total base score, then original immediate rank.
4. Require distinct draft and tile placements within each species channel.
5. Deduplicate all additions against the frozen frontier.
6. Keep the five-turn cutoff, R8 shared determinizations, pattern-aware
   continuation, terminal acting-seat base score, public redetermination, and
   seeded tie handling unchanged.

Two candidates per species is the smallest frontier that provides placement
redundancy. It adds at most ten actions and introduces no learned weights,
species coefficients, hidden information, or score shaping.

The strategy ID is
`late-wildlife-diverse-policy-improvement-v1-t5-r8-k8-h6-b8-w2-m4`.

## Required Tests

- the expanded frontier is a superset of K8+H6+B8;
- every visible wildlife species has candidate coverage;
- expansion is bounded by ten additional actions;
- zero wildlife width is rejected;
- the hybrid is exactly pattern-aware before cutoff;
- terminal actions are legal, deterministic, and replayable;
- the strategy ID records every frozen configuration field.

## Experiment

The mandatory sequential smoke uses seed 27299 and must finish within 30
treatment seconds. A passing implementation runs seeds 27300-27309 and
requires:

- paired mean delta at least +0.5;
- Bear delta at least 0.0;
- total wildlife delta at least +0.25;
- aggregate Elk, Salmon, Hawk, and Fox delta at least 0.0;
- habitat delta at least -0.5;
- Nature Token delta at least -1.0;
- treatment runtime at most 15 seconds per game;
- treatment P90 decision latency at most 1.5 seconds.

Only a passing pilot may run the frozen 50-game confirmation on seeds
27400-27449. Confirmation requires a paired 95% confidence interval lower
bound above zero and the same category and runtime guardrails.

No wildlife width, cutoff, base frontier, determinization count, continuation
policy, target, or tie-rule tuning is permitted between stages.

## Result

The smoke passed at +1.250 and 10.901 treatment seconds, with 522 ms P90
decision latency. The frozen ten-game pilot then scored 91.750 against 91.200:

- paired delta +0.550, 95% CI -0.123 to +1.223, record 8-0-2;
- Bear +1.625;
- Elk -0.475, Salmon -0.650, Hawk -0.200, Fox -0.200;
- total wildlife +0.100;
- aggregate non-Bear wildlife -1.525;
- habitat +0.500;
- Nature Tokens -0.050;
- treatment runtime 10.976 seconds per game;
- treatment P90 decision latency 574 ms.

The score, Bear, habitat, token, and runtime gates passed. Total wildlife
missed its +0.250 gate by 0.150, and aggregate non-Bear wildlife missed its
nonnegative gate by 1.525. No confirmation was permitted.

The candidate superset did not repair allocation. It exposed more actions to
the same finite-sample maximization and produced the same Bear-for-other-
species exchange seen in earlier broadening experiments. The next mechanism
should control selection error in the terminal evaluator rather than add
another frontier channel.
