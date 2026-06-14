# Determinized Lookahead V2 K4/R4/D4

Status: superseded on 2026-06-10 by
`determinized-lookahead-v2-k8-r4-d4`.

## Policy

The strategy resolves public market events, keeps the four best exact
immediate-score actions, evaluates them across four common
information-consistent hidden states, plays four greedy plies, and selects the
largest mean acting-seat leaf score.

Search randomness is derived independently for each decision from the public
game seed, completed turn, and acting seat. The CLI, stateless API, replay
service, and tests therefore execute the same policy entry point.

## Evidence

Pilot, seeds 20300-20319:

- greedy mean: 86.513
- treatment mean: 89.738
- paired delta: +3.225
- 95% CI: [2.099, 4.351]
- record: 18-0-2

Confirmation, seeds 20400-20449:

- greedy mean: 86.880
- treatment mean: 89.435
- paired delta: +2.555
- 95% CI: [1.915, 3.195]
- record: 44-1-5
- original paired wall time: 232.23 seconds

## Exact Runtime Optimization

Native sampling showed that repeated full-board scoring, especially habitat
connectivity, dominated candidate generation and left most worker threads idle
outside the four retained actions. The implementation now:

- evaluates candidate-by-determinization jobs in one indexed Rayon workload;
- carries the placed tile and wildlife into the evaluation callback; and
- rescans only affected habitat terrains and wildlife cards.

Fox is always rescored after any wildlife placement. Salmon D and Hawk D are
also treated as cross-species dependencies. An exhaustive regression test
compares delta and full rescoring for every A-D card family on all legal
midgame candidates.

The optimized implementation reproduced every score and action outcome in the
20-game pilot exactly. Wall time fell from 153.461 to 41.115 seconds, a 3.73x
speedup, without changing the policy.

It also reproduced the complete 50-game confirmation suite exactly in 111.129
seconds instead of 232.233 seconds, a 2.09x end-to-end speedup under the load
present during that run.

Command:

```bash
target/release/cascadia-v2 lookahead-compare \
  --games 50 \
  --first-seed 20400 \
  --baseline greedy \
  --candidates 4 \
  --determinizations 4 \
  --greedy-plies 4 \
  --output docs/v2/reports/determinized-lookahead-v2-k4-r4-d4-vs-greedy-50.json
```

The confirmed mean remains 10.565 points below the primary target. Bear,
habitat, Fox, and retained Nature Tokens improved; Salmon, Elk, and Hawk remain
the clearest longer-horizon opportunity.
