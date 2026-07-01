# ADR 0032: Perfect-Information Focal Beam

Status: multi-turn continuation mechanism confirmed on 2026-06-11;
diagnostic only and not promoted.

## Context

One-step perfect-information policy improvement reached 93.975 with the W2
frontier. The exact W2 treatment beat the exact base frontier by 1.350, proving
candidate recall matters, but remained below 97. Its action values still
finished every candidate with frozen pattern-aware continuation, so future
focal decisions inherited the same allocation policy the oracle was trying to
improve.

The Fox-only public transfer then tied promoted strong exactly. Candidate
channels are not the next public lever until value identification improves.
Before building another model, directly measure whether joint multi-turn
planning raises the hidden-information diagnostic ceiling.

## Decision

Compare two focal-seat hidden-information diagnostics against three frozen
pattern-aware opponents:

- baseline: the exact K8+H6+B8+W2 one-step oracle from ADR 0030;
- treatment: the same oracle before the final five personal turns, then a
  beam planner that jointly optimizes all remaining focal decisions.

For each root candidate, the beam preserves the true hidden stack and bag,
plays opponents with the frozen pattern-aware policy, expands the exact W2
frontier at every future focal turn, and retains the best 16 focal states by
the frozen pattern heuristic. Final selection uses exact focal terminal base
score. Common deterministic continuation randomness is cloned across root
candidates and beam branches.

This is diagnostic only. Hidden information may not enter product play,
training inputs, or final strength claims.

## Frozen Protocol

- Rules: canonical four-player AAAAA, no habitat bonuses.
- Runtime smoke: seed 29599.
- Pilot: seeds 29600-29609.
- Four focal-seat rotations per seed against three pattern-aware seats.
- Frontier: K8+H6+B8+W2+M4.
- Beam width: 16.
- Beam activation: final five personal turns.
- Local CPU only.

The treatment must complete the smoke within 180 seconds per four-rotation
block. A passing smoke authorizes the fixed pilot.

Interpretation:

- paired beam-minus-one-step gain at least +0.50: multi-turn continuation
  planning is a material independent lever;
- treatment mean at least 97: joint focal planning materially raises the
  current diagnostic ceiling;
- treatment mean at least 100: the frontier contains target-level focal play
  under exact future information;
- gain below +0.25: this beam formulation does not repair continuation.

Report mean, paired interval, score categories, latency, runtime, record,
determinism, and complete provenance.

## Required Tests

- configuration rejects zero beam width, zero wildlife width, and invalid
  activation turns;
- strategy ID records every frozen parameter;
- width is enforced independently per root candidate;
- root actions and all simulated actions remain legal;
- deterministic ranking and complete match replay hold;
- with one focal turn remaining, beam selection matches one-step exact
  continuation under the same seed;
- the CLI comparison rotates the focal seat and reports explicit score blocks.

## Result

The runtime smoke passed at 163.852 treatment seconds for four focal-seat
rotations and tied the one-step oracle at 95.250.

Across seeds 29600-29609 and 40 focal scores per strategy:

- exact W2 one-step mean: 92.900;
- exact W2 focal-beam mean: 93.650;
- paired gain: +0.750, 95% CI `[+0.400,+1.100]`;
- game-block record: 9-1-0;
- treatment runtime: 89.306 seconds per four-rotation block;
- treatment P90 focal decision latency: 3,252 ms.

The gain decomposed into Habitat +0.250, Wildlife +0.175, and Nature Tokens
+0.325. Wildlife deltas were Bear +0.400, Elk +0.675, Salmon -0.775, Hawk
+0.325, and Fox -0.450.

The gain exceeds the frozen +0.50 materiality threshold with a strictly
positive interval. Jointly optimizing several focal turns is therefore an
independent continuation lever. The treatment mean remains below 97 and
6.350 points below target, so this beam does not establish target-level value
inside the current frontier.

The next public-state learning target should estimate multi-turn
counterfactual advantage or continuation quality. Distilling one-step R8
choices, relabeling narrow trajectory outcomes, and widening candidates
without a stronger continuation representation are all closed by prior
evidence.
