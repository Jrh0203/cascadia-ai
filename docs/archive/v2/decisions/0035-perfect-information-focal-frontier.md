# ADR 0035: Perfect-Information Focal Frontier

Status: rejected after pilot on 2026-06-11.

## Context

Exact W2 candidate breadth improved the one-step oracle by 1.350, establishing
structural wildlife action recall as real signal. Exact final-five beam
planning independently improved W2 one-step play by 0.750. Portfolio-preserving
beam retention then changed only one of ten seed blocks, so reshuffling the
same width-16 children is closed.

The next bounded diagnostic combines the two constructive findings: test
whether the focal beam is still constrained by admitting only two distinct
actions per wildlife species.

## Decision

Compare two perfect-information scalar focal beams against three frozen
pattern-aware opponents:

- baseline: K8+H6+B8+W2+M4 frontier;
- treatment: K8+H6+B8+W4+M4 frontier.

Both use width 16, the original scalar pattern heuristic, final-five
activation, exact hidden state, exact terminal scoring, identical opponent
policy, and common deterministic continuation randomness.

This is diagnostic only and non-promotable.

## Frozen Protocol

- Rules: canonical four-player AAAAA, no habitat bonuses.
- Runtime smoke: seed 30499.
- Pilot: seeds 30500-30509.
- Four focal-seat rotations per seed.
- Beam width: 16.
- Beam activation: final five personal turns.
- Local CPU only.

The W4 treatment must complete the smoke within 180 seconds per four-rotation
block.

Interpretation:

- paired W4-minus-W2 gain at least +0.25: focal action recall remains a
  material exact-search limit;
- gain below +0.10: widening this wildlife frontier is not the missing
  continuation mechanism;
- treatment mean at least 97: the wider frontier materially raises the
  diagnostic ceiling;
- treatment mean at least 100: target-level focal play exists under exact
  future information and this bounded search.

Report the paired interval, all score categories, runtime, latency, record,
determinism, and provenance. No frontier width, beam width, cutoff, heuristic,
opponent, continuation, or threshold tuning is permitted between stages.

## Required Tests

- configuration names W2 and W4 distinctly and rejects zero wildlife width;
- both strategies are deterministic, legal, and replayable;
- final-turn action selection remains exact terminal scoring;
- baseline and treatment receive explicit focal score blocks;
- the CLI report captures complete typed configuration and provenance.

## Runtime Optimization

The first smoke reproduced a promising +3.250 score but missed the runtime
gate: W4 took 232.275 seconds. Before any pilot, a behavior-preserving engine
pass replaced repeated full frontier-record clones with stable index
orderings and let scalar selectors compute only the tied maximum instead of
sorting every evaluated candidate.

The optimized rerun reproduced every score, category delta, and paired result
exactly while reducing W4 to 163.404 seconds, a 29.7% improvement. Seeded
selection equivalence, frontier-reference equality, the complete simulation
and search suites, formatting, and strict Clippy all passed. The pilot was
then authorized.

## Result

The passing optimized smoke scored 95.500 versus W2 at 92.250, but that
single-seed +3.250 did not replicate.

Across seeds 30500-30509:

- W2 beam mean: 94.000;
- W4 beam mean: 93.075;
- paired delta: -0.925, 95% CI `[-2.426,+0.576]`;
- record: 4-0-6;
- treatment runtime: 92.552 seconds per four-seat block;
- treatment P90 decision latency: 3,403 ms.

W4 gained 1.425 Fox and 0.050 Hawk, but lost 1.250 Bear, 0.500 Elk, 0.275
Salmon, 0.550 total wildlife, and 0.400 habitat. Nature Tokens gained 0.025.

The treatment failed the primary score threshold, remained below 97, and
never reached 100 in the pilot. No confirmation is permitted.

Unrestricted W4 expansion under a fixed width-16 scalar beam crowds out
valuable Bear and habitat branches. Wildlife action recall remains real from
the earlier exact W2 result, but widening both root and future layers is not
the solution. A narrower controlled successor must isolate root candidate
recall from future-layer beam competition.
