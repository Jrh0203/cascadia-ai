# ADR 0021: Start Terminal Search One Turn Earlier

Status: rejected on 2026-06-11 after the registered confirmation.

## Context

The final-four-turn R8 hybrid produced a low-variance +0.475 paired gain with a
strictly positive 95% interval, interactive runtime, and positive total
wildlife and habitat. It missed advancement by 0.025 total points and lost
0.100 Bear. Full-game R8 had gained 1.750 Bear in its qualification.

Card A Bear value requires both setup and completion. Beginning exact terminal
search with only four personal turns remaining may improve conversion of
existing plans while arriving too late to create an additional pair.

## Decision

Run one adaptive successor on a fully disjoint seed suite:

1. Change only the cutoff from four to five remaining personal turns.
2. Preserve exact pattern-aware RNG streams before the cutoff.
3. Keep R8, K8+H6+B8, public redetermination, common random numbers,
   pattern-aware continuation, terminal scoring, and tie handling frozen.
4. Do not sweep any other cutoff or tune from the smoke result.

The strategy ID is
`late-terminal-policy-improvement-v1-t5-r8-k8-h6-b8-m4`.

## Experiment

The mandatory sequential smoke uses seed 26899 and must finish within 90
treatment seconds. A passing implementation runs seeds 26900-26909 and
requires:

- paired mean delta at least +0.5;
- Bear delta at least +0.25;
- total wildlife delta at least 0.0;
- habitat delta at least -0.5;
- Nature Token delta at least -1.0;
- treatment runtime at most 10 seconds per game;
- treatment P90 decision latency at most 1.5 seconds.

Only a passing pilot may run the frozen 50-game confirmation on seeds
27000-27049. Confirmation requires a paired 95% confidence interval lower
bound above zero and the same mechanism and runtime guardrails.

## Result

The one-game smoke passed. The first pilot execution produced the same score
result later reported below but exceeded the runtime gate. Two
behavior-preserving optimizations were then completed before interpreting the
pilot:

- wildlife placements already legal on the pre-action board are scored once
  per decision and reused exactly across candidate tiles;
- the candidate generator extends that cached set only with placements made
  legal by the candidate tile.

All A-D wildlife cards, every legal action in representative states, and the
ten-seed pattern-aware reference remained exactly score-identical. The
optimized frozen pilot then passed every gate:

- baseline 92.100, treatment 93.100, paired delta +1.000;
- 95% CI +0.574 to +1.426, record 9-0-1;
- Bear +0.750, total wildlife +0.700, habitat +0.300, Tokens +0.000;
- treatment runtime 7.506 seconds per game;
- treatment P90 decision latency 387 ms.

That authorized the disjoint 50-game confirmation. It scored 92.135 against
91.710:

- paired delta +0.425, 95% CI +0.198 to +0.652, record 35-6-9;
- Bear +0.200;
- total wildlife +0.145;
- habitat +0.340;
- Nature Tokens -0.060;
- treatment runtime 7.530 seconds per game;
- treatment P90 decision latency 382 ms.

The primary confidence-interval, total-wildlife, habitat, token, and runtime
requirements passed. The frozen confirmation reused the pilot's mechanism
guardrails, so Bear required at least +0.250. The observed +0.200 missed that
gate by 0.050. The policy is therefore rejected and is not promoted despite
the statistically positive total-score result.

The experiment establishes that five-turn terminal search has reproducible
positive conversion value at interactive latency. It does not establish the
preregistered Bear mechanism strongly enough for promotion. A successor must
change the candidate-information mechanism, not sweep the cutoff again.
