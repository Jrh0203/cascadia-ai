# Canonical Benchmark Protocol

Protocol ID: `cascadia-aaaaa-4p-base-v1`

## Primary Metric

Four-player symmetric `AAAAA` play. Habitat bonuses are calculated for
diagnostics but excluded from the primary score.

Every seat uses the identical strategy implementation, configuration, model,
pre-move policy, and latency/search budget. A game contributes four independent
seat-score observations and one correlated game block.

## Rules

- Four players.
- Twenty turns per player.
- Cards: Bear A, Elk A, Salmon A, Hawk A, Fox A.
- Standard tile and wildlife bags.
- Full legal nature-token behavior.
- Free overflow replacement and paid mulligans are part of the strategy's
  complete `TurnAction`.
- Base score = wildlife + largest habitats + remaining nature tokens.
- Habitat-majority bonus is not included in the primary score.

## Information Boundary

Strategies receive:

- all public boards,
- public market,
- public counts of remaining tiles/tokens where rules expose them,
- current player, turn, cards, and legal actions.

Strategies do not receive future bag order. Search creates sampled futures from
the public belief state using its own deterministic RNG stream.

## Randomness

Each game seed deterministically derives separate streams for:

- game setup and bag order,
- strategy randomness for each seat,
- stochastic search futures for each seat.

Comparisons use the same game seeds and derived stream identities. Adding
logging or changing thread scheduling must not change game results.

## Seed Suites

Seed suites are named, versioned artifacts.

- `dev`: routine iteration; may be inspected repeatedly.
- `test`: confirmation of candidates selected on `dev`.
- `validation`: sealed final strength suite; never used for training or tuning.

Dataset splits and benchmark suites use distinct domain-separated hashes so a
numeric seed cannot silently cross roles.

## Development Comparisons

- Use paired game/seat observations.
- State the hypothesis and promotion threshold before execution.
- Report both absolute results and paired delta.
- Report games, seat scores, mean, SD, SE, 95% CI, P10/P50/P90, min/max, and
  habitat/wildlife/token breakdown.
- Report wall time, decisions, mean and percentile decision latency, hardware,
  commit, dirty digest, and full typed configuration.
- Small smoke runs can reject crashes or large regressions but cannot promote.

Default promotion stages:

1. 20-game smoke.
2. 100-game development comparison.
3. 250-game test comparison.
4. 1,000-game held-out validation for champion claims.

Sequential stopping rules, if used, must be specified before the run.

## Final 100-Point Claim

A claim of mean base score at least 100.0 requires:

- at least 1,000 held-out games / 4,000 seat scores,
- a 95% confidence interval reported at the game-block level,
- no overlap between training/tuning seeds and validation seeds,
- verification that all seats used the same complete policy,
- verification that habitat bonuses were excluded,
- verification that no strategy observed future bag order,
- replayable result artifacts,
- a paired comparison against the independently reproduced strongest v1
  baseline.

## V1 Reproduction

V1 results are not imported from reports. The v2 runner will host a v1 adapter
that invokes one complete v1 turn policy consistently for every seat. Until
that adapter exists, current CLI and `mce_perf_bench` outputs are diagnostic
only because their pre-move and opponent semantics differ.

