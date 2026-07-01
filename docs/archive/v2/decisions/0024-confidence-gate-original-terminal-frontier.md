# ADR 0024: Confidence-Gate the Original Terminal Frontier

Status: historically accepted on 2026-06-11; superseded and demoted by
ADR 0068 on 2026-06-12 after canonical redetermination requalification.

## Context

The terminal-search evidence now forms three corners of a controlled matrix:

| Frontier | Selection | Score | Non-Bear wildlife |
|---|---|---:|---:|
| K8+H6+B8 | maximum R8 mean | +0.425 confirmed | -0.055 |
| K8+H6+B8+W2 | maximum R8 mean | +0.550 pilot | -1.525 |
| K8+H6+B8+W2 | c90 paired lower bound | +0.825 pilot | -0.800 |

Confidence gating improved both score and allocation on the expanded frontier,
but W2 still failed the non-Bear guardrail. The untested fourth corner applies
the exact same confidence rule to the original frontier, which had nearly
neutral non-Bear behavior in confirmation.

## Decision

Run that ablation without changing any other boundary:

1. Use the original K8+H6+B8 frontier with no W2 additions.
2. Use the final-five cutoff and eight shared public-information
   determinizations.
3. Compute the exact pattern-aware action as anchor.
4. Admit a challenger only when its one-sided 90% paired Student-t lower bound
   is strictly positive, using `t(0.90, 7) = 1.4149239276488585`.
5. Select the admitted action with the largest lower bound; otherwise play the
   anchor exactly.
6. Keep pattern-aware continuation, terminal acting-seat base score, market
   prelude, and seeded ties frozen.

The strategy ID is
`late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`.

## Required Tests

- configuration rejects any sample count other than eight;
- the strategy ID records the original frontier and c90 rule;
- complete matches are legal, deterministic, and replayable;
- the shared paired-bound tests from ADR 0023 remain green;
- pre-cutoff actions exactly match pattern-aware.

## Experiment

The mandatory sequential smoke uses seed 27899 and must finish within 30
treatment seconds. A passing implementation runs seeds 27900-27909 and
requires:

- paired mean delta at least +0.25;
- Bear delta at least 0.0;
- total wildlife delta at least 0.0;
- aggregate Elk, Salmon, Hawk, and Fox delta at least 0.0;
- habitat delta at least -0.5;
- Nature Token delta at least -1.0;
- treatment runtime at most 12 seconds per game;
- treatment P90 decision latency at most 1.2 seconds.

Only a passing pilot may run the frozen 50-game confirmation on seeds
28000-28049. Confirmation requires a paired 95% confidence interval lower
bound above zero and the same category and runtime guardrails.

No confidence level, cutoff, frontier width, sample count, continuation
policy, target, or tie-rule tuning is permitted between stages.

## Result

All registered stages passed without changing the frozen method.

| Stage | Seeds | Paired delta | 95% CI | Bear | Wildlife | Non-Bear | Habitat | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pilot | 27900-27909 | +0.500 | [-0.027, +1.027] | +0.225 | +0.450 | +0.225 | +0.250 | -0.200 |
| Confirmation | 28000-28049 | +0.420 | [+0.179, +0.661] | +0.080 | +0.115 | +0.035 | +0.365 | -0.060 |

The confirmation treatment scored 91.915 against 91.495 for pattern-aware,
with a 28-9-13 record. It ran in 6.995 seconds per complete game with 362 ms
P90, 881 ms P99, and 2.253 seconds maximum decision latency. Every score,
allocation, runtime, and latency gate passed.

This is the first v2 terminal-search policy to survive both a disjoint
confirmation and all frozen mechanism guardrails. It is promoted as the local
`strong` product tier. The promotion is deliberately narrower than a general
claim about confidence gating: the accepted policy is exactly the original
K8+H6+B8 frontier, final-five cutoff, R8 shared samples, pattern-aware
continuation, and c90 paired lower-bound rule recorded above.
