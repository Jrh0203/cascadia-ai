# Determinized Lookahead V1 K4/R4/D4

Status: confirmed on 2026-06-10, then superseded before product promotion.

This file preserves the completed evidence for the superseded RNG schedule.

## Hypothesis

Two-turn public-information lookahead can recognize short wildlife
commitments, market effects, and Nature Token value that exact immediate-score
greedy misses.

## Algorithm

For each move:

1. resolve the optional free three-of-a-kind replacement and observe the
   resulting public market,
2. enumerate every legal action and retain the four highest immediate
   base-score candidates,
3. sample four common information-consistent hidden states,
4. apply each candidate and play four additional plies with exact greedy
   policies,
5. choose the candidate with the highest mean acting-seat leaf base score.

Hidden sampling shuffles only the unseen tile/exclusion pool and wildlife bag.
It preserves every public fact and uses the same samples for every candidate.

## Registered Evidence

Pilot, seeds 20000-20009:

- greedy mean: 86.450
- lookahead mean: 89.250
- paired delta: +2.800
- 95% CI: [1.035, 4.565]
- record: 8-0-2

Confirmation, seeds 20100-20149:

- greedy mean: 86.505
- lookahead mean: 89.815
- paired delta: +3.310
- 95% CI: [2.610, 4.010]
- record: 45-1-4
- elapsed: 281.43 seconds for 50 paired games

Command:

```bash
target/release/cascadia-v2 lookahead-compare \
  --games 50 \
  --first-seed 20100 \
  --baseline greedy \
  --candidates 4 \
  --determinizations 4 \
  --greedy-plies 4 \
  --output docs/archive/v2/reports/determinized-lookahead-v1-k4-r4-d4-vs-greedy-50.json
```

## Interpretation

The largest category gain was Bear (+1.065), followed by habitat (+0.930
combined) and retained Nature Tokens (+1.375). Elk and Hawk each regressed
slightly, so deeper or better leaf evaluation should target those longer
commitments.

The algorithmic result remains valid, but the mutable per-seat search RNG could
not be reproduced by the stateless web API without replaying prior searches.
`determinized-lookahead-v2` replaces it with an explicit per-decision seed
schedule and must pass its own confirmation before product promotion.
