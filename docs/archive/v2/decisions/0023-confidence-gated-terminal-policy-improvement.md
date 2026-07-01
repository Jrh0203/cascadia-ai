# ADR 0023: Confidence-Gate Terminal Policy Improvement

Status: rejected on 2026-06-11 after the registered pilot.

## Context

Final-five R8 search has reproducible positive total-score signal. Expanding
its frontier also passed a ten-game score gate, but converted additional value
almost entirely into Bear while losing 1.525 Elk+Salmon+Hawk+Fox.

Both operators select the maximum of noisy eight-sample candidate means.
Adding candidates increases the chance that a high-variance action wins by
sampling error. Shared determinizations already provide paired outcomes
against the promoted pattern-aware action, so unsupported deviations can be
rejected directly.

## Decision

Test one conservative policy-improvement rule:

1. Preserve the final-five cutoff and the wildlife-diverse
   K8+H6+B8+W2 frontier.
2. Compute the exact pattern-aware action first and retain it as the anchor.
3. Evaluate every candidate and the anchor on the same eight hidden-state
   redeterminizations with frozen pattern-aware continuation.
4. For each challenger, compute its eight paired terminal-score advantages
   over the anchor.
5. Admit a challenger only when the one-sided 90% Student-t lower confidence
   bound is strictly above zero. With eight samples, the fixed critical value
   is `t(0.90, 7) = 1.4149239276488585`.
6. Select the admitted action with the largest lower bound; fall back exactly
   to the pattern-aware anchor when none qualifies.

The confidence level is fixed before gameplay and will not be swept. The
strategy consumes the same pattern-aware RNG draw at every terminal decision,
so fallback behavior preserves the promoted policy exactly.

The strategy ID is
`late-conservative-policy-improvement-v1-t5-r8-k8-h6-b8-w2-m4-c90`.

## Required Tests

- configuration rejects any determinization count other than eight;
- the strategy ID records the cutoff, frontier, and confidence rule;
- a consistent paired advantage passes the bound;
- a positive mean caused by one outlier fails the bound;
- complete matches are legal, deterministic, and replayable;
- pre-cutoff actions exactly match pattern-aware.

## Experiment

The mandatory sequential smoke uses seed 27599 and must finish within 30
treatment seconds. A passing implementation runs seeds 27600-27609 and
requires:

- paired mean delta at least +0.25;
- Bear delta at least 0.0;
- total wildlife delta at least 0.0;
- aggregate Elk, Salmon, Hawk, and Fox delta at least 0.0;
- habitat delta at least -0.5;
- Nature Token delta at least -1.0;
- treatment runtime at most 15 seconds per game;
- treatment P90 decision latency at most 1.5 seconds.

Only a passing pilot may run the frozen 50-game confirmation on seeds
27700-27749. Confirmation requires a paired 95% confidence interval lower
bound above zero and the same category and runtime guardrails.

No confidence level, cutoff, frontier, sample count, continuation policy,
target, or tie-rule tuning is permitted between stages.

## Result

The smoke passed its runtime gate at 11.198 treatment seconds and 589 ms P90
decision latency. The frozen ten-game pilot then scored 92.600 against 91.775:

- paired delta +0.825, 95% CI +0.452 to +1.198, record 9-0-1;
- Bear +1.150;
- Elk -0.325, Salmon -0.600, Hawk +0.100, Fox +0.025;
- total wildlife +0.350;
- aggregate non-Bear wildlife -0.800;
- habitat +0.575;
- Nature Tokens -0.100;
- treatment runtime 8.951 seconds per game;
- treatment P90 decision latency 484 ms.

The primary score, Bear, total-wildlife, habitat, token, and runtime gates
passed. Aggregate non-Bear wildlife failed its nonnegative gate by 0.800, so
no confirmation was permitted.

The confidence gate reduced the wildlife-diverse frontier's non-Bear loss from
1.525 to 0.800 while increasing its score gain from 0.550 to 0.825. This is
evidence that finite-sample maximization was part of the failure. It did not
remove the allocation tradeoff, which remains associated with the W2
candidate expansion. The remaining clean ablation is the same fixed
confidence rule on the original K8+H6+B8 frontier.
