# ADR 0036: Perfect-Information Root-Diverse Beam

Status: rejected after pilot on 2026-06-11.

## Context

Exact W2 established material wildlife candidate recall, and final-five beam
planning established material continuation value. W4 at every beam layer then
regressed by 0.925 despite gaining 1.425 Fox. The coupled treatment changed
both current action recall and every future layer's branching pressure under
a fixed width-16 beam.

The next controlled diagnostic separates those effects.

## Decision

Compare two perfect-information scalar focal beams against three frozen
pattern-aware opponents:

- baseline: W2 at the current decision and every future focal layer;
- treatment: W4 only at the current decision, then W2 at every future focal
  layer.

Both use K8+H6+B8+M4 anchors, width 16, final-five activation, the original
scalar heuristic, exact hidden state, exact terminal scoring, common
deterministic continuation randomness, and identical opponents. Before the
cutoff, both use the same exact W2 one-step policy.

This is diagnostic only and non-promotable.

## Frozen Protocol

- Rules: canonical four-player AAAAA, no habitat bonuses.
- Runtime smoke: seed 30699.
- Pilot: seeds 30700-30709.
- Four focal-seat rotations per seed.
- Root wildlife width: baseline W2, treatment W4.
- Future wildlife width: W2 for both.
- Beam width: 16.
- Beam activation: final five personal turns.
- Local CPU only.

Treatment must complete the smoke within 180 seconds per four-seat block.

Interpretation:

- paired root-W4-minus-W2 gain at least +0.25: root candidate recall remains
  material after future crowding is removed;
- gain below +0.10: W4 root recall is not the missing mechanism;
- treatment mean at least 97: root-only breadth raises the diagnostic ceiling;
- treatment mean at least 100: target-level focal play exists under exact
  future information and this bounded search.

No root width, future width, beam width, cutoff, heuristic, continuation,
opponent, or threshold tuning is permitted between stages.

## Required Tests

- strategy identity records root and future wildlife widths separately;
- zero root or future width is rejected;
- before cutoff, treatment exactly matches W2 one-step;
- at cutoff, only root candidate coverage differs;
- complete play is deterministic, legal, and replayable;
- CLI reports explicit focal score blocks with typed provenance.

## Result

The runtime smoke tied at 93.500 and passed at 121.296 treatment seconds.

Across seeds 30700-30709, W4 root scored 94.625 versus W2 at 94.550:
+0.075 with 95% CI `[-0.030,+0.180]` and a 2-8-0 record. Treatment runtime
was 79.557 seconds per four-seat block with 3,525 ms P90 latency.

Habitat and total wildlife were flat. Nature Tokens gained 0.075. Salmon and
Hawk improved 0.150 and 0.250, while Elk and Fox fell 0.225 and 0.175; Bear
was unchanged.

The gain missed the +0.10 mechanism threshold, eight blocks tied, and the
treatment remained below 97. Candidate-width experiments are closed; the
next exact question is beam capacity or continuation-state evaluation.
