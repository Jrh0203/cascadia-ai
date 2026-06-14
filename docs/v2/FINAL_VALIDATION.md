# Final Strength Validation

The final strength suite is a resumable, distributed, 1,000-game benchmark on
the canonical four-player AAAAA rules engine. Habitat bonuses are excluded.
Every game contributes four treatment seat scores and one correlated game
block for confidence intervals.

## Frozen Seed Domain

Game indices `0-999` are transformed through `DatasetSplit::Final`. They are
therefore cryptographically domain-separated from every train, validation, and
test game, even when another split uses the same numeric index.

The benchmark report records both the public game index and the derived
32-byte `GameSeed`. Aggregation fails on a missing, duplicated, or extra index.

## Current Factual Reference

Until a fresh v2 model clears its complete validation and gameplay gates, the
strongest canonical-engine reference is:

`canonical-action-legacy-exact-mlx-v1-k32-r600-lmr-no-paid-prelude`

It uses the canonical v2 rules, legality, scoring, and public-information
boundary. Every neural forward runs through MLX. Its historical parameters
make it a research reference, not a promoted final v2 model.

Each final game is paired against:

`late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`

The generated report also includes the independently reproduced v1
cross-engine reference. That comparison is explicitly absolute rather than
paired.

## Three-Node Layout

Use one process per Apple GPU. The frozen 1,000-game layout is:

| Node | First index | Games | Last index |
|---|---:|---:|---:|
| john1 | 0 | 334 | 333 |
| john2 | 334 | 333 | 666 |
| john3 | 667 | 333 | 999 |

Every completed game has an atomic JSON report, checksum-bound metadata,
stdout/stderr logs, host identity, source revision, command, environment, and
input fingerprints. Restarting the same command validates and skips complete
games. Partial or drifted evidence is rejected.

Example shard:

```bash
make final-strength-shard \
  FINAL_STRENGTH_OUTPUT_DIR=artifacts/final-strength/john1 \
  FINAL_STRENGTH_FIRST_GAME_INDEX=0 \
  FINAL_STRENGTH_GAMES=334
```

After retrieving all three complete shard directories:

```bash
make final-strength-aggregate \
  FINAL_STRENGTH_SHARDS="artifacts/final-strength/john1 artifacts/final-strength/john2 artifacts/final-strength/john3" \
  FINAL_STRENGTH_FIRST_GAME_INDEX=0 \
  FINAL_STRENGTH_GAMES=1000
```

## Integrity Gates

Aggregation requires:

- exact coverage of final indices `0-999`;
- one source revision and one executable/model/weight fingerprint set;
- exactly four seat scores and 80 decision timings per strategy per game;
- internally consistent score decomposition;
- the canonical protocol and frozen strategy identities;
- all bridge/runtime smoke gates passing;
- clean MLX service shutdown after every game.

The final report contains the mean, game-block and seat-score standard
deviations, standard error, 95% confidence interval, P10/P50/P90, score
breakdown, decision latency, paired result, host distribution, provenance, and
the explicit 100-point verdict.

## Infrastructure Rehearsal

A clean john3 checkout at revision `6b4a43a` completed one R600 game at
final-domain index `999999`, outside the reserved final suite. It exercised all
80 decisions for each paired strategy, wrote and recovered the raw four-seat
record, verified one input fingerprint set and source revision, shut down MLX
cleanly, and passed aggregation.

The 94.5 one-game treatment score is not strength evidence. The machine-readable
rehearsal is
[`final-strength-infrastructure-smoke.json`](reports/final-strength-infrastructure-smoke.json).
