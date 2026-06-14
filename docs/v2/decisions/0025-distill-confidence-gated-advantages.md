# ADR 0025: Distill Confidence-Gated Terminal Advantages

Status: rejected on validation on 2026-06-11.

## Context

The promoted strong policy is a statistically confirmed improvement, but its
R8 terminal search costs 362 ms at P90. Earlier MLX rankers failed because
they tried to reproduce a noisy total ordering over many nearly identical
afterstates. Strong does not need that ordering. It compares every challenger
with one exact pattern-aware anchor under shared samples and changes the move
only when the paired c90 lower bound is positive.

That paired decision is the smallest faithful learning target and preserves
the mechanism that passed confirmation.

## Decision

Build `conservative-advantage-v1` as a separate, versioned pipeline:

1. Play fresh four-player AAAAA trajectories with the promoted strong policy.
2. Record only final-five decisions.
3. For every non-anchor challenger, store both hidden-state-safe observable
   action afterstates, the paired mean advantage, standard error, c90 lower
   bound, and whether strong selected that challenger.
4. Train a shared-weight MLX pair encoder to regress the lower bound in score
   points.
5. At inference, retain the exact pattern-aware anchor unless the model's
   highest predicted lower bound is positive.

The model never receives hidden refill or stack order. Train, validation, test,
gameplay, and final seed domains remain disjoint.

## Fixed Data And Model

- Train: 128 games, train indices 0-127.
- Validation: 32 games, validation indices 0-31.
- Untouched test: 32 games, test indices 0-31.
- One-game atomic resumable shards.
- Exact strong teacher: final five turns, R8, K8+H6+B8, M4, c90.
- Architecture: hidden 96, four heads, two board blocks, one market block,
  pairwise shared encoder, regression trunk.
- AdamW, learning rate 1e-4, weight decay 1e-4, at most 20 epochs, validation
  patience five.

No test data may be read for architecture, threshold, loss, or hyperparameter
selection.

## Advancement Gates

The selected validation checkpoint must improve mean squared lower-bound error
over the zero predictor and satisfy:

- mean teacher policy regret at most 0.20 points;
- exact anchor-or-challenger action agreement at least 65%;
- anchor false-positive rate at most 20%;
- exact selected-challenger recall at least 35%;
- candidate lower-bound correlation at least 0.50.

The untouched test split must pass the same gates. Only then may a ten-game
paired gameplay pilot run on seeds 28500-28509. The distilled final-five policy
must be no worse than -0.25 against strong, beat pattern-aware by at least
+0.25, lose no more than 0.5 total wildlife or habitat, lose no more than one
Nature Token, and run in at most one second per complete game.

Only a passing pilot may authorize a disjoint 50-game confirmation. Extending
the learned policy earlier than the final-five boundary is a separate
experiment and cannot be inferred from distillation fidelity.

## Result

Collection completed exactly as registered:

- train: 128 games, 2,560 groups, 38,381 challengers;
- validation: 32 games, 640 groups, 10,051 challengers;
- untouched test: 32 games, 640 groups, 9,813 challengers.

The fixed MLX run stopped after 17 epochs and 4,352 optimizer steps at
validation patience. Epoch 12 was the selected checkpoint:

| Metric | Best validation | Gate |
|---|---:|---:|
| Mean squared error | 0.966878 | below zero predictor |
| Zero-predictor MSE | 4.789431 | reference |
| Mean policy regret | 0.095716 | <= 0.20 |
| Exact policy agreement | 0.765625 | >= 0.65 |
| Anchor false-positive rate | 0.004073 | <= 0.20 |
| Selected-challenger recall | 0.006711 | >= 0.35 |
| Lower-bound correlation | 0.760583 | >= 0.50 |

The model learned a useful continuous value signal but almost always retained
the anchor. A validation-only threshold diagnostic showed this was not merely
calibration: even without a threshold, the selected challenger was the
highest-ranked challenger in only 16.8% of challenger-selected groups.

The selected-challenger recall gate failed decisively. The untouched test
labels were not evaluated, no gameplay pilot ran, and no model was promoted.
The next experiment changes the learning problem to balanced groupwise policy
learning rather than tuning this regressor's threshold.
