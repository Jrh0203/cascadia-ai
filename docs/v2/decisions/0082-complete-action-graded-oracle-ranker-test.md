# ADR 0082: Complete-Action Graded Oracle Ranker Sealed Test

Status: closed unopened after ADR 0081 validation failed.

Date: 2026-06-15

## Purpose

ADR 0081 selects one of three preregistered MLX replicas using only seeds
`61003`, `61007`, and `61011`. This ADR freezes the only authorized test
before any validation metric is known. If any ADR 0081 gate fails, the test
remains unopened and this ADR closes without evaluation.

## Frozen Test

- Source games: `61004`, `61008`, and `61012`.
- Exactly 240 complete decisions.
- Source experiment and grouped schema: byte-identical to ADR 0081.
- Checkpoint: the single ADR 0081 replica selected before authorization.
- Test execution: once, on john1, with an independent replay on john3.
- Model, weights, score calibration, thresholds, candidate ordering, feature
  decoder, and evaluation code are frozen before authorization.

Before the first model reads a test group, the supervisor must write
`test-authorization.json` containing:

- every passing ADR 0081 validation gate and report checksum;
- selected checkpoint, tensor, model-config, and source-code checksums;
- test dataset and decoder checksums;
- proof that no model-evaluation output exists for the test split;
- authorization timestamp and exact test command.

The evaluator rejects missing or late authorization, a changed checkpoint,
source mismatch, schema mismatch, incomplete group, non-finite score, hidden
state feature, or any attempt to evaluate more than once.

## Test Gates

The frozen checkpoint passes only if:

- all source, replay, split, checkpoint, authorization, and checksum integrity
  gates pass;
- top-64 recall of the R4800 winner is strictly greater than 98%;
- retained mean R4800 regret is strictly less than 0.15 points;
- early, middle, and late top-64 recall are each at least 97%;
- early, middle, and late retained mean regret are each below 0.20 points;
- no action-family subset with at least 20 groups falls below 95% top-64
  recall or exceeds 0.25 retained mean regret;
- all 240 groups score every canonical legal action exactly once;
- john1 and john3 produce identical rankings, metrics, and selected hashes.

Report the complete ADR 0081 metric set even when a gate fails. Test metrics
cannot alter the checkpoint, thresholds, or gameplay treatment.

Passing authorizes only ADR 0083 inference integration and gameplay. Failure
permanently rejects this exact experiment without retry, alternate test split,
ensemble, calibration, or gameplay.

## Closure

ADR 0081 selected the john2 replica, but its 73.33% top-64
R4800-winner recall failed the frozen >98% validation gate along with every
phase and subset recall gate. No `test-authorization.json` was created, no
model read a sealed-test group, and no test evaluation output exists. This ADR
is permanently closed unopened for this experiment.

## Maximum Compute

One test evaluation and one independent replay. No retraining, validation
resume, additional seed, model selection, gameplay, or external compute is
authorized.
