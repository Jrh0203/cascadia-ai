# ADR 0031: Fox-Focused Terminal Frontier

Status: rejected after pilot on 2026-06-11.

## Context

The perfect-information W2 diagnostic improved the exact base frontier by
1.350 points with a strictly positive interval. Fox supplied +1.775, while
Bear, Elk, Salmon, and Hawk summed to -0.525 and habitat fell 0.500.

Earlier public-information all-species W2 search was positive overall but
failed its non-Bear allocation guardrail. The promoted base-frontier c90
operator avoided that damage and confirmed +0.420 over 50 games. The clean
transfer test is therefore not another all-species expansion: it is adding
only the candidate channel implicated by the exact diagnostic.

## Decision

Compare promoted strong against one treatment that changes only terminal root
candidate recall:

- baseline:
  `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`;
- treatment:
  the same final-five R8 c90 operator plus at most two distinct Fox-drafting
  candidates ranked by exact resulting Fox score, exact base score, and
  immediate rank.

The treatment preserves every K8+H6+B8 action. It does not add Bear, Elk,
Salmon, or Hawk coverage. Public hidden-state redetermination, eight shared
samples, pattern-aware continuation, the one-sided 90% paired lower bound,
anchor fallback, market prelude, and seeded tie handling remain identical.

## Frozen Protocol

- Rules: canonical four-player AAAAA, no habitat bonuses.
- Runtime smoke: seed 29299.
- Pilot: seeds 29300-29309.
- Confirmation, only after every pilot gate passes: seeds 29400-29449.
- Final personal turns: 5.
- Determinizations: 8.
- Base frontier: K8+H6+B8+M4.
- Fox coverage: F2.
- Local CPU only.

The smoke treatment must complete within 20 seconds per game.

Pilot gates:

- paired treatment-minus-strong mean at least +0.25;
- Fox delta at least +0.25;
- total wildlife delta at least 0.0;
- aggregate Bear+Elk+Salmon+Hawk delta at least -0.50;
- habitat delta at least -0.50;
- Nature Token delta at least -1.0;
- treatment runtime at most 12 seconds per game;
- treatment P90 decision latency at most 1.0 second.

Confirmation requires a paired 95% confidence-interval lower bound above zero,
Fox delta at least +0.10, total wildlife delta at least 0.0, aggregate
Bear+Elk+Salmon+Hawk delta at least -0.25, habitat delta at least -0.50, and
the same runtime limits.

No species, width, confidence level, cutoff, sample count, continuation,
target, or threshold tuning is permitted between stages.

## Required Tests

- the focused frontier is a bounded superset of K8+H6+B8;
- only the selected species receives an explicit coverage channel;
- zero focused width is rejected;
- the strategy ID records species and every frozen parameter;
- treatment decisions are legal, deterministic, and replayable;
- before the cutoff, treatment and strong both reproduce pattern-aware;
- the CLI report compares strong directly with the focused treatment and
  retains complete provenance.

## Result

The runtime smoke tied strong exactly and passed at 7.422 treatment seconds
with 250 ms P90 decision latency.

The frozen pilot then produced an exact score-block tie on every seed:

| Metric | Strong | Strong + Fox F2 | Delta |
|---|---:|---:|---:|
| Mean base score | 92.150 | 92.150 | 0.000 |
| Habitat | 28.500 | 28.500 | 0.000 |
| Wildlife | 59.300 | 59.275 | -0.025 |
| Nature Tokens | 4.350 | 4.375 | +0.025 |

The record was 0-10-0. Fox gained only 0.050 against the required 0.250.
Bear changed -0.100, Salmon +0.100, Hawk -0.075, and Elk was flat, for
aggregate non-Fox wildlife -0.075. Treatment runtime was 5.525 seconds per
game and P90 decision latency was 344.233 ms.

The implementation and runtime gates passed, but both the +0.25 score gate and
the +0.25 Fox gate failed. No confirmation is permitted.

The exact-information diagnostic and this public-information null separate
candidate availability from value identification: useful Fox actions exist,
but the frozen R8 c90 estimator cannot identify net-positive use of them on
strong trajectories. Wider candidate channels are closed until a
decision-local public-state evaluator improves.
