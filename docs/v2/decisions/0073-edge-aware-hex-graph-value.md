# ADR 0073: Edge-Aware Hex Graph Score-To-Go

Status: rejected on fresh validation on 2026-06-13. The sealed test and
gameplay domains remain unopened.

## Context

The fresh V2 neural models encode all four public boards, the market, phase,
per-seat wildlife counts, habitat sizes, and Nature Tokens. Opponent
information is therefore present. The common board encoder is nevertheless a
permutation-invariant set transformer followed by mean/max pooling. It receives
axial coordinates and tile rotation as ordinary scalar/categorical features,
but it has no explicit neighborhood operator, oriented terrain-edge relation,
or connected-component inductive bias.

That omission is unusually expensive in Cascadia. Habitat score is defined by
matching oriented edges; Bear, Salmon, Hawk, Fox, and Elk all depend on local
or multi-hop tile relations. The rejected H6 score-to-go model reached 2.569
reconstructed-final MAE but only 0.397 correlation. Its target was learnable,
yet the model mostly learned phase and average remaining workload.

Message-passing networks explicitly update entities from their graph
neighbors, and graph-network formulations treat edge attributes as first-class
relations. This experiment follows those ideas without importing a library or
historical model:

- [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)
- [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
- [Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ)

The local hypothesis is narrower than any literature claim: exact Cascadia
hex adjacency and terrain-edge relations will improve held-out value
discrimination over the existing generic set encoder.

## Decision

Add MLX architecture `edge-aware-hex-score-to-go-v2`:

1. Reuse the immutable `compact-entity-v2` records and signed decomposed
   score-to-go targets. No hidden state, bag order, legacy feature vector, or
   historical weight file becomes an input.
2. Reconstruct exact directed adjacency from integer axial coordinates for
   each of the four relative-seat boards.
3. Reconstruct each tile's terrain on all six oriented edges from terrain A,
   optional terrain B, and rotation. Every directed neighbor relation carries
   a five-value matching-terrain edge attribute.
4. Project each tile to width 96 and apply four residual message-passing
   blocks. A shared message MLP receives the neighboring hidden state,
   six-way direction, and matching-terrain edge attribute. Messages are summed
   over legal neighbors; padded entities cannot send or receive messages.
5. Preserve separate relative-seat embeddings. Pool every board independently,
   then combine the four board summaries with the existing market and global
   branches.
6. Predict the same eleven signed score-to-go components. Keep the existing
   normalized component loss and total-score consistency loss.
7. Add a `0.25`-weighted pairwise logistic term over positions from the same
   game and personal-turn round. The target is the difference in final base
   score between the two acting-seat perspectives, at a fixed two-point
   temperature. This trains opponent-relative discrimination while retaining
   absolute component calibration.
8. Apply deterministic independent random 60-degree board rotations during
   training. Rotate coordinates, tile rotations, and no targets. Validation is
   unaugmented.

This is a fresh V2 model. The existing entity-set checkpoint is evaluated as a
frozen baseline on the fresh validation domain, but it is not used to
initialize the graph model.

## Frozen Experiment

- Train corpus: immutable 256-game H6 train dataset
  `score-to-go-habitat-candidate-lookahead-v1-k8-h6-r4-d4-train-0`,
  indices 0-255.
- Fresh validation: validation split indices 64,000-64,063, 64 H6 games.
- Sealed test: test split indices 64,000-64,063, 64 H6 games. It remains
  unopened unless every validation gate passes.
- Implementation-only smoke: train and validation index 9,992, one game each.
- Teacher trajectory: frozen `habitat-candidate-lookahead-v1-k8-h6-r4-d4`.
- Model: width 96, four graph blocks, one market attention block, feed-forward
  multiplier three, eleven output components.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Training: batch 256, seed 20260624, at most 30 epochs, validation patience
  six, checkpoint every 500 steps.
- Selection metric: validation within-round pairwise log loss plus `0.1`
  times reconstructed-final total MAE.
- No warm start, architecture sweep, width/depth selection, loss-weight
  tuning, alternate validation domain, retry, or post-hoc blend.

Before training, evaluate the frozen selected
`entity-set-score-to-go-v1` checkpoint on the fresh validation split. The graph
checkpoint advances only if every validation gate passes:

- dataset, target-identity, source, checkpoint, resume, and device checks pass;
- reconstructed-final total correlation is at least 0.50 and improves over
  the frozen set baseline by at least 0.05;
- reconstructed-final total MAE is at most 3.00 and does not regress from the
  frozen set baseline by more than 0.10;
- within-round pairwise accuracy improves over the frozen set baseline by at
  least three percentage points;
- within-round pairwise log loss improves over the frozen set baseline;
- no wildlife-component final MAE regresses from the set baseline by more than
  0.50 point;
- all predictions and gradients remain finite;
- warmed batch-256 Apple-GPU inference is no slower than 25 milliseconds per
  position-equivalent batch item.

Passing validation authorizes the single sealed 64-game test collection. Test
must preserve correlation at least 0.48, MAE at most 3.15, and directional
improvement over the frozen set baseline in correlation, pairwise accuracy,
and pairwise log loss, with the same wildlife guardrail.

Passing test authorizes a separately preregistered leaf-search experiment.
This ADR does not authorize gameplay, promotion, threshold changes, or
additional training based on validation or test results.

## Maximum Compute

One one-game implementation smoke, one fresh 64-game H6 validation collection,
one baseline evaluation, one Apple-GPU training run of at most 30 epochs, one
inference benchmark, and conditionally one sealed 64-game test collection and
evaluation. All work is local on the Apple M4. No external compute, second
training run, hyperparameter sweep, validation retry, or gameplay is
authorized.

## Implementation Qualification

`make score-to-go-hexgraph-smoke` passed before the fresh validation domain was
opened:

- Rust collected independent train and validation games at implementation-only
  index 9,992, 80 positions each;
- both checksummed datasets passed the independent score-to-go validator and
  exact `current + residual = final` identity;
- exact directed adjacency, reverse directions, matching terrain edges,
  rotation, masking, finite forward/loss, and padded-entity isolation pass in
  the MLX test suite;
- all 96 Python tests pass, with clean Ruff lint and formatting;
- one Metal epoch completed one optimizer step in 0.098 seconds;
- the selected checkpoint reloaded with exact manifest checks;
- validation evaluated all 120 within-round pairs and produced finite
  component, total, calibration, ranking, and selection metrics.

The one-game metrics are implementation evidence only and cannot select the
substantive model. The preserved report is
`docs/v2/reports/edge-aware-hex-score-to-go-v2-implementation-smoke.json`.

## Result

The single authorized training run stopped after epoch 21 under the frozen
patience rule and selected epoch 15. It completed 5,376 optimizer steps in
824.97 seconds on `Device(gpu, 0)`.

On 64 fresh validation games and 5,120 positions, the frozen entity-set
baseline and edge-aware graph model measured:

| Metric | Set baseline | Hex graph | Delta | Gate |
|---|---:|---:|---:|---|
| Reconstructed-final correlation | 0.3933 | 0.3417 | -0.0516 | fail |
| Reconstructed-final MAE | 2.5415 | 2.7982 | +0.2567 | fail |
| Within-round pairwise accuracy | 64.7406% | 65.3890% | +0.6484 pp | fail |
| Within-round pairwise log loss | 0.7628 | 0.7296 | -0.0331 | pass |

All five wildlife-component MAE guardrails passed. Predictions and gradients
remained finite, dataset and checkpoint integrity passed, and the sealed test
dataset was never collected. Warmed batch-256 Metal inference reached 88.70
milliseconds P90 in total, 0.346 milliseconds per position and approximately
2,908 positions per second, passing the performance gate by a wide margin.

The exact graph inductive bias improved the soft ranking loss but did not
improve outcome discrimination. It regressed both final-score correlation and
MAE, and its pairwise accuracy gain was less than one percentage point rather
than the required three. This is a valid quality rejection, not an
implementation or runtime failure.

Geometry alone is closed on this single-trajectory H6 target. Future value
work must improve the target information by estimating counterfactual
expected return under multiple public-information futures, with explicit
supply and opponent-demand conditioning, before another architecture
comparison is justified.
