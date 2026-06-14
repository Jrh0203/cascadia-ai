# ADR 0029: Perfect-Information Frontier Bound

Status: one-step target-level hypothesis rejected on 2026-06-11.

## Context

The promoted strong policy scores 91.915, the independently reproduced v1
reference scores 95.895, and the research target is 100. Repeated public-state
search, candidate, and neural changes have not established whether the current
K8+H6+B8 frontier already contains 100-point play.

Before another model or heuristic is built, isolate two possible limits:

1. the frontier and frozen pattern continuation do not contain enough value;
2. sufficient actions exist, but public-information uncertainty and value
   estimation fail to identify them.

## Decision

Implement a diagnostic-only perfect-information policy-improvement oracle.
For one focal seat at a time it:

- applies the same automatic market prelude as pattern-aware;
- constructs the exact K8+H6+B8 pattern frontier;
- preserves the canonical hidden stack and wildlife bag rather than
  redetermining them;
- applies each candidate to a clone and finishes the complete game with the
  frozen pattern-aware policy for all seats;
- uses one common deterministic continuation RNG seed across candidates;
- ranks by the acting seat's exact final base score, then immediate score.

The policy is explicitly non-promotable because it observes hidden future
draws. It exists only to measure a frontier/continuation upper bound.

Each numeric seed is a four-seat block. Four games rotate the focal oracle
through seats 0-3 while the other three seats remain frozen pattern-aware.
The baseline is the corresponding four scores from one all-pattern-aware game.
This keeps actual continuation behavior aligned with candidate evaluation.

## Frozen Protocol

- Baseline: `pattern-aware-v1-k8-h6-b8-m4`.
- Treatment:
  one focal `perfect-information-pattern-oracle-v1-k8-h6-b8-m4` seat against
  three pattern-aware seats, rotated through all four seats.
- Rules: canonical four-player AAAAA, no habitat bonuses.
- Runtime smoke: seed 28899.
- Pilot: ten paired sequential games, seeds 28900-28909.
- Full-game continuation after every candidate; no depth cutoff.
- Local CPU only.

The full configuration must complete the smoke within 120 seconds. If it does,
run the fixed pilot and report mean, paired interval, category breakdown,
latency, runtime, and record.

Interpretation is diagnostic, not promotional:

- treatment mean at least 100 or paired gain at least +5: current frontier
  contains target-level value; prioritize uncertainty/value learning;
- treatment mean below 97: current frontier or continuation is a material
  ceiling; prioritize wider structural candidates or a stronger continuation;
- intermediate results retain both hypotheses.

No hidden-information strategy may enter the API, web product, production
strategy enum, model data, or final benchmark.

## Smoke Correction

The first implementation smoke ran the oracle symmetrically and completed in
9.823 seconds, proving runtime viability. It scored 89.0 versus strong at 93.0,
but that comparison is methodologically invalid for the stated bound: after
the chosen move, real future opponents also used the oracle, while candidate
rollouts assumed frozen pattern-aware continuation.

This was detected before the frozen pilot. The protocol above replaces that
symmetric smoke with focal-seat rotations and introduces an explicit
score-block report type so aggregated scores are never attached to a replay
from a different game.

## Result

The corrected runtime smoke passed at 15.536 seconds for all four focal-seat
rotations. The ten-seed pilot then produced 40 focal seat scores:

- pattern-aware baseline mean: 91.375;
- focal perfect-information oracle mean: 93.150;
- paired gain: +1.775, 95% CI `[+0.299,+3.251]`;
- game-block record: 8-0-2;
- P90 focal decision latency: 274.609 ms;
- four-rotation wall time: 12.457 seconds per seed block.

The category change was Habitat +0.375, Nature Tokens -0.275, and Wildlife
+1.675. Wildlife allocation was extreme: Bear +11.325, Elk -4.975, Salmon
-0.050, Hawk -4.050, and Fox -0.575.

The treatment failed the preregistered 97-point diagnostic boundary and is
6.850 points below the target. Exact future draws and exact terminal outcomes
therefore do not make one-step greedy improvement over this frontier
target-level under the frozen pattern continuation.

This is not a mathematical upper bound on every policy using the frontier.
It is one exact policy-improvement step with pattern-aware continuation. The
result rejects the stated target-level hypothesis and identifies continuation
allocation as a material bottleneck. A controlled wildlife-diverse oracle
frontier can now separate missing structural candidates from that continuation
failure without stochastic winner's curse.
