# ADR 0019: Anchor the Conditioned Commitment Premium

Status: rejected on 2026-06-11 after the registered pilot.

## Context

Two independent rules-derived experiments found positive cross-turn signal:

- phase-capped optimistic commitment scored +0.650 and Bear +1.900, but lost
  0.950 non-Bear wildlife and 0.650 habitat;
- exact first-rotation competition scored +0.875, Bear +1.275, habitat +0.400,
  and Nature Tokens +0.725, but lost 1.525 non-Bear wildlife.

Both replaced the promoted one-turn opportunity value with a new two-turn
scalar. The first-turn revaluation is the common route by which a useful Bear
commitment signal can distort the already confirmed global allocation.

## Decision

Retain the exact promoted K8+H6+B8 frontier, immediate base score, and
one-turn optimistic opportunity. Add only the incremental value that exact
first-rotation competition assigns to a second future personal turn.

For each candidate observable public afterstate:

1. Compute `O1`, the promoted one-turn optimistic wildlife opportunity.
2. Compute `C1`, the exact opponent-conditioned expected opportunity with one
   future personal turn.
3. Compute `C2`, the same exact first-rotation expectation with two future
   personal turns.
4. Rank by `immediate_base + O1 + (C2 - C1)`.

The same terminal market distribution is reused for `C1` and `C2`. Since every
two-turn per-species continuation contains its one-turn immediate gain,
`C2 >= C1` apart from floating-point roundoff. No learned weight, blend
coefficient, species constant, or hidden order is introduced.

The acting-seat horizon is capped by exact remaining turns. With only one
future personal turn, `C2 == C1`, so the complete ranking must be byte-for-byte
equal to promoted pattern-aware. On the final personal turn all opportunity is
zero.

## Required Tests

- one-future-turn ranking exactly equals pattern-aware;
- final-personal-turn opportunity is exactly zero;
- conditioned premium is finite and non-negative;
- hidden wildlife-bag redetermination leaves the complete ranking unchanged;
- selected actions are legal and seeded reproducibly;
- all exact market replacement tests from ADR 0018 remain green.

## Experiment

Strategy ID:
`pattern-portfolio-v1-k8-h6-b8-m4-t2-conditioned-premium`.

Scheduling affects timing but not deterministic scores, so runtime is measured
with sequential games to represent an interactive local session without CPU
contention from nine simultaneous matches.

The mandatory smoke uses seed 26299 and must finish within five treatment
seconds. A passing implementation runs seeds 26300-26309 and requires:

- paired mean delta at least +0.5;
- Bear delta at least +0.5;
- total wildlife delta at least 0.0;
- aggregate non-Bear wildlife delta at least -0.5;
- habitat delta at least -0.5;
- treatment runtime at most five seconds per game.

Only a passing pilot may run the frozen 50-game confirmation on seeds
26400-26449. Confirmation requires a paired 95% confidence interval lower
bound above zero and the same category and runtime guardrails.

## Result

The smoke passed at 2.805 treatment seconds with a diagnostic +3.0 paired
result. The frozen ten-game pilot scored 92.575 versus 92.550:

- paired delta +0.025, 95% CI -1.291 to +1.341, record 4-0-6;
- Bear -0.575;
- total wildlife -0.075;
- aggregate non-Bear wildlife +0.500;
- habitat +0.025;
- Nature Tokens +0.075;
- treatment runtime 2.866 seconds per game.

Anchoring achieved its allocation objective: the 1.525-point non-Bear loss
from full competition became a 0.500-point gain, with flat habitat and passing
runtime. It also removed the strength mechanism. The experiment failed the
score, Bear, and total-wildlife gates, so no confirmation was run.
