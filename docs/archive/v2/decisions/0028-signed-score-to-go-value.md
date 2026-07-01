# ADR 0028: Learn Signed Score To Go

Status: rejected at validation on 2026-06-11.

## Context

The first H6 value model used the final decomposed score as the target for all
80 positions in a game. It achieved 2.538 final-total MAE but only 0.212
correlation, failing before leaf-search gameplay. Stronger trajectories did
not create enough target variation for action ranking.

That target forces the network to reconstruct exact current score and future
gain simultaneously. It also repeats one narrow game-level label across every
phase. The next controlled experiment isolates target semantics.

## Decision

Collect the same H6 K8+H6/R4/D4 train and validation seed domains into a new
fixed-width dataset:

- input: the existing hidden-state-safe `compact-entity-v2` position record;
- audit target: exact final eleven-component base score;
- audit baseline: exact current eleven-component base score;
- learned target: signed `final - current` score-to-go components.

The schema is `signed-score-to-go-components-v1`. Residuals are signed 16-bit
integers and are checked against current plus final components on every read.

Train the same hidden-96, four-head, two-board-block, one-market-block entity
encoder from scratch, with an eleven-output signed linear head identified as
`entity-set-score-to-go-v1`. The only semantic change is that the network
predicts score to go. Validation reconstructs final components by adding exact
current score and reports both residual and reconstructed-final metrics.

## Frozen Protocol

- Train: 256 H6 games, train indices 0-255.
- Validation: 64 H6 games, validation indices 0-63.
- One-game resumable shards.
- AdamW learning rate 3e-4, weight decay 1e-4.
- Batch 256, at most 20 epochs.
- Select the lowest reconstructed-final total MAE.

The model advances to a separately gated leaf-search pilot only if:

- reconstructed-final total correlation is at least 0.50;
- reconstructed-final total MAE is at most 4.0;
- no wildlife component final MAE exceeds the rejected final-target model by
  more than 1.0 point;
- dataset round trips, checksums, target identities, checkpoint resume, and
  MLX GPU training all pass.

No gameplay result may be inferred from validation alone.

## Implementation Evidence

The implementation smoke and a disjoint two-game collector probe passed before
substantive collection:

- Rust round trips preserve negative residuals and the exact
  `current + residual = final` identity.
- Python decoding, entity encoding, signed output, checkpoint writing, and a
  one-epoch Apple GPU run pass.
- Parallel collection is independent of shard granularity: the probe generated
  two games concurrently while committing two ordered, checksummed one-game
  shards, then passed full dataset validation.
- Rust tests and strict Clippy pass for the data and CLI crates; focused Python
  tests and Ruff pass.

The rejected final-target wildlife MAEs are frozen as Bear 3.902832, Elk
3.259204, Salmon 3.639360, Hawk 3.301609, and Fox 3.514257. The corresponding
score-to-go advancement ceilings are 4.902832, 4.259204, 4.639360, 4.301609,
and 4.514257.

## Result

The frozen collection completed with 20,480 train positions and 5,120
validation positions across 320 checksummed one-game shards. Exact target
identity held for every record. Validation final totals span 82-99, while
score-to-go totals fall from a mean 86.70 at turn zero to 5.06 at turn 76.
Negative component residuals occur only for spent Nature Tokens.

The selected epoch-13 checkpoint produced:

- reconstructed-final total MAE 2.568601;
- reconstructed-final total correlation 0.397451;
- residual-total correlation 0.991700;
- wildlife MAE 3.918660, 3.054139, 3.394859, 3.140786, and 3.000673.

Every MAE and component gate passed, but reconstructed-final correlation missed
the required 0.50. The maximum correlation at any epoch was 0.414201, so the
failure is not an artifact of selecting by MAE.

Predicting score to go cleanly learns phase and remaining workload, but adding
current score back cancels two large correlated quantities and exposes the
same narrow game-outcome signal that limited final-target learning. The target
change improved correlation from 0.211986 to 0.397451 but did not create a
qualified leaf evaluator. No model was promoted and no gameplay ran.
