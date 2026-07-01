# ADR 0033: Public Focal-Beam Teacher

Status: rejected after pilot on 2026-06-11.

## Context

The exact final-five focal beam improved the exact W2 one-step oracle by
0.750 with a strictly positive interval. It establishes multi-turn
continuation as real signal. Hidden information is diagnostic-only, so the
next step is to test whether a bounded public redetermination policy can
recover enough of that signal to qualify as a stronger teacher.

Promoted strong uses one-step R8 terminal continuation on the base frontier
with a c90 anchor fallback. Earlier all-species W2 one-step search failed
allocation guardrails. This experiment changes the continuation mechanism,
not merely candidate breadth: every sampled candidate is followed by joint
future focal planning.

## Decision

Compare promoted strong with one public-information treatment:

- final five personal turns;
- K8+H6+B8+W2 root and future focal frontiers;
- four shared public hidden-state redeterminizations;
- independent width-four focal beam per root candidate and determinization;
- frozen pattern-aware opponents and beam heuristic;
- paired candidate-minus-pattern-anchor outcomes;
- one-sided 90% Student-t lower bound with
  `t(0.90, 3) = 1.6377443536962095`;
- select the largest positive lower bound, otherwise play the exact
  pattern-aware anchor.

Before the cutoff, treatment is exactly pattern-aware, matching strong.
Actual hidden stack and bag order are never evaluated.

## Frozen Protocol

- Rules: canonical four-player AAAAA, no habitat bonuses.
- Runtime smoke: seed 29799.
- Pilot: seeds 29800-29809.
- Confirmation, only after every pilot gate passes: seeds 29900-29949.
- Final personal turns: 5.
- Determinizations: 4.
- Beam width: 4.
- Frontier: K8+H6+B8+W2+M4.
- Confidence rule: one-sided paired c90.
- Local CPU only.

Smoke treatment runtime must be at most 150 seconds per complete symmetric
game.

Pilot gates:

- paired treatment-minus-strong mean at least +0.25;
- total wildlife delta at least 0.0;
- aggregate Elk+Salmon+Hawk+Fox delta at least -0.50;
- habitat delta at least -0.50;
- Nature Token delta at least -1.0;
- treatment runtime at most 120 seconds per game;
- treatment P90 decision latency at most 10 seconds.

Confirmation requires a paired 95% confidence-interval lower bound above
zero, total wildlife at least 0.0, aggregate non-Bear wildlife at least -0.25,
habitat at least -0.50, and the same runtime limits.

No sample count, beam width, frontier, confidence level, cutoff, continuation,
heuristic, or threshold tuning is permitted between stages.

Passing confirmation qualifies this policy as a research teacher and permits
a separately preregistered MLX continuation-advantage dataset. It does not
automatically promote a product strategy because of its research-tier
latency.

## Required Tests

- configuration rejects zero work and any unsupported sample count;
- c90 uses the exact registered four-sample critical value;
- every sample starts from a public redetermination;
- candidate and anchor share identical sample seeds;
- root candidates retain independent beam budgets;
- before cutoff, treatment matches pattern-aware exactly;
- complete games are deterministic, legal, and replayable;
- CLI reports strong directly against treatment with complete provenance.

## Result

The runtime smoke passed at 88.208 treatment seconds with 5,898 ms P90
decision latency, but regressed 1.000 point on its single seed.

Across seeds 29800-29809:

- promoted strong mean: 92.925;
- public focal-beam mean: 92.850;
- paired delta: -0.075, 95% CI `[-0.565,+0.415]`;
- record: 5-2-3;
- treatment runtime: 114.312 seconds per game;
- treatment P90 decision latency: 4,920 ms.

Habitat gained 0.050 and Nature Tokens fell 0.100. Bear gained 0.475, while
aggregate Elk+Salmon+Hawk+Fox fell exactly 0.500; total wildlife fell 0.025.

The treatment passed runtime, latency, habitat, token, and non-Bear boundary
gates. It failed the primary +0.25 score gate by 0.325 and the nonnegative
total-wildlife gate by 0.025. No confirmation or MLX dataset is permitted.

The exact beam's continuation signal does not transfer through four-sample,
width-four public redetermination. More sampling of this bounded teacher is
not justified by gameplay strength. The next exact diagnostic should improve
how beam capacity preserves competing wildlife portfolios before another
public recovery attempt.
