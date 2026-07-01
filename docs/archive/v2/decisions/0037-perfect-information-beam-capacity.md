# ADR 0037: Perfect-Information Beam Capacity

Status: rejected after pilot on 2026-06-11.

## Context

Exact final-five B16 planning adds real continuation value. Category portfolio
retention was null, W4 at every layer regressed, and W4 only at the root was
also null. Candidate breadth is closed. The remaining bounded search question
is whether W2 continuations are lost because width 16 is insufficient.

## Decision

Compare exact scalar focal beams with W2 everywhere:

- baseline: width 16;
- treatment: width 32.

Everything else is frozen: K8+H6+B8+W2+M4, final-five activation, exact hidden
state, scalar pattern heuristic, exact terminal scoring, deterministic
continuations, and three pattern-aware opponents.

## Frozen Protocol

- Runtime smoke: seed 30899.
- Pilot: seeds 30900-30909.
- Four focal-seat rotations per seed.
- Treatment runtime gate: 180 seconds per four-seat block.
- Local CPU only.

Gain at least +0.25 means beam capacity is material; gain below +0.10 rejects
capacity as the missing mechanism. Treatment mean at least 97 raises the
diagnostic ceiling; at least 100 reaches target under exact information.

No width, frontier, cutoff, heuristic, continuation, opponent, or threshold
tuning is permitted between stages.

## Result

The smoke tied at 91.500 and passed at 91.069 treatment seconds.

Across seeds 30900-30909, B32 scored 94.100 versus B16 at 94.075: +0.025
with 95% CI `[-0.024,+0.074]` and a 1-9-0 record. Treatment runtime was
95.548 seconds per four-seat block with 3,056 ms P90 latency.

Habitat and Nature Tokens were flat. Salmon fell 0.075 and Fox gained 0.100;
all other wildlife categories were unchanged.

The gain is below the +0.10 mechanism threshold, nine blocks tied, and the
treatment remained below 97. Beam capacity is closed. The remaining exact
limit is continuation-state evaluation.
