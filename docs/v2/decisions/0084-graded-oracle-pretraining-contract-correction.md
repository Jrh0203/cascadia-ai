# ADR 0084: Graded-Oracle Pretraining Contract Correction

Status: locked before model initialization, training, or validation evaluation.

Date: 2026-06-16

Experiment ID: `complete-action-graded-oracle-ranker-v1`

## Context

Lossless conversion exposed two implementation-contract defects before any
model was trained.

First, the workspace exporter and standalone source verifier compiled
`serde_json` with different float parser feature sets. Two source summaries
therefore differed by one ULP after parsing even though their immutable game
records were unchanged.

Second, ADR 0081 called 8,192 the maximum number of complete action rows in a
batch. The train and validation corpus contains ten indivisible complete
groups above that threshold. The largest has 10,854 actions. Splitting,
truncating, or sampling a group would violate the complete-action objective.
No model or Python evaluator inspected the sealed test split while diagnosing
this contradiction.

## Decision

1. Enable `serde_json/float_roundtrip` for the entire Rust workspace, matching
   the standalone verifier.
2. Permit only decimal-spelling normalization of derived summary fields. The
   immutable game and decision records may not change, and both Rust
   validators must accept every repaired shard.
3. Retain 8,192 padded action rows as the multi-group packing target.
4. Admit an indivisible group above 8,192 only as a singleton batch.
5. Freeze a 16,384-action hard ceiling for any singleton group. A larger group
   rejects the run; it is never split, sampled, or truncated.

The 16,384 ceiling is the next power-of-two safety bound above the 10,854
maximum observed in train and validation. It was selected without reading
sealed-test groups in Python.

## Identity Amendment

The source experiment, rules, games, decisions, action records, split,
architecture, loss, optimizer, seeds, validation gates, and test protocol are
unchanged.

- Source manifest SHA-256 remains
  `c5e568644e1f7d2d11eed7b6099778853ecd4898cd586f31b5edaddd566fdca5`.
- Source manifest BLAKE3 remains
  `2751d33ae1da1d9cad4355555c29f988d69caf93c96158fd8d6f7d6499415ca7`.
- Corrected collection index SHA-256 is
  `d574cd07b13c69efb186fa206849955035429d5828cae598195bf1453cdc7fb4`.
- Corrected collection index BLAKE3 is
  `cdca37a5eb030ba4f5461e4673b2cac0ea0e27951b2caedf1e5a94940f89c426`.
- Frozen conversion exporter SHA-256 is
  `4b9fd643f5cd016a42cc40e6fc7b47a4500f0a2406d06f706ad61e52fe0bcabd`.

Repair receipts live beside seeds `61002` and `61010`. The invalid first
conversion launch and its logs are preserved under
`artifacts/experiments/complete-action-graded-oracle-ranker-v1/invalid-launch-serde-contract/`.
All partial datasets from that launch were discarded before the successful
three-host conversion.

## Consequences

ADR 0081 remains authoritative except where this correction explicitly
supersedes its source-index identity and 8,192-row wording. Training may begin
only after Rust and Python validate the corrected train and validation
artifacts and a maximum-width forward/backward smoke passes under the frozen
16,384 singleton ceiling.
