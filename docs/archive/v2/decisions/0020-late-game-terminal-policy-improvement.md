# ADR 0020: Spend Terminal Search Only When It Is Cheap

Status: rejected on 2026-06-11 after the registered pilot.

## Context

R8 terminal policy improvement qualified as a constructive teacher at +1.333
paired over three games, with gains in Bear, total wildlife, and habitat. Its
original implementation cost 185.878 seconds per game. Behavior-preserving
profiling work has reduced the reference seed from 273.884 to roughly 80
seconds while preserving every score, but full-game use is still not an
interactive policy.

The cost is strongly phase dependent. Early candidates require dozens of
pattern-aware continuation plies; late candidates require only a few. The
promoted pattern-aware policy is already strongest in routine early drafting,
while exact terminal simulation can directly price endgame conversion.

## Decision

Test one frozen hybrid policy:

1. Use the promoted pattern-aware policy while the acting player has more than
   four personal turns remaining.
2. Use the qualified R8 terminal policy-improvement operator for the acting
   player's final four turns.
3. Preserve the exact pattern-aware per-seat RNG streams before the cutoff so
   baseline and treatment trajectories are identical until terminal search
   first intervenes.
4. Keep the K8+H6+B8 frontier, R8 shared determinizations, public-information
   redetermination, pattern-aware continuation, and terminal base-score target
   unchanged.

No phase weight, species coefficient, learned value, reduced determinization
count, or post-smoke cutoff tuning is permitted.

## Required Tests

- invalid cutoffs are rejected;
- the hybrid selects exactly the recorded pattern-aware action before cutoff;
- it selects exactly the R8 terminal action at and after cutoff;
- complete matches are legal, deterministic, and replayable;
- the strategy ID records the cutoff and all terminal configuration.

## Experiment

Strategy ID:
`late-terminal-policy-improvement-v1-t4-r8-k8-h6-b8-m4`.

The mandatory sequential smoke uses seed 26599 and must finish within 60
treatment seconds. A passing implementation runs seeds 26600-26609 and
requires:

- paired mean delta at least +0.5;
- Bear delta at least 0.0;
- total wildlife delta at least 0.0;
- habitat delta at least -0.5;
- Nature Token delta at least -1.0;
- treatment runtime at most 20 seconds per game;
- treatment P90 decision latency at most 3 seconds.

Only a passing pilot may run the frozen 50-game confirmation on seeds
26700-26749. Confirmation requires a paired 95% confidence interval lower
bound above zero and the same category and runtime guardrails.

## Result

The smoke passed at 5.272 treatment seconds with a +1.000 paired result and a
262 ms P90 decision latency. The frozen ten-game pilot then scored 92.250
against 91.775:

- paired delta +0.475, 95% CI +0.197 to +0.753, record 8-1-1;
- Bear -0.100;
- total wildlife +0.225;
- habitat +0.225;
- Nature Tokens +0.025;
- treatment runtime 5.895 seconds per game;
- treatment P90 decision latency 315 ms.

The low-variance positive interval proves that terminal conversion has useful
late-game signal, and the runtime is interactive. The experiment nevertheless
missed the frozen +0.500 score gate by 0.025 and the nonnegative Bear gate by
0.100. No confirmation was permitted. Four remaining personal turns begins
too late to recover the Bear mechanism observed in full-game R8.
