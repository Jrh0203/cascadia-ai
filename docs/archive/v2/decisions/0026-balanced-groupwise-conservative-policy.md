# ADR 0026: Balanced Groupwise Conservative Policy

Status: rejected on validation on 2026-06-11.

## Context

`conservative-advantage-v1` regressed paired c90 lower bounds accurately but
failed to reproduce strong's sparse policy changes. On validation, only 4.0%
of challengers had positive lower bounds and only 23.3% of decision groups
selected a challenger. The best regressor achieved 0.761 lower-bound
correlation but recovered only 0.7% of selected challengers. Removing the
zero threshold raised the ceiling to just 16.8% exact challenger recall.

This is a groupwise decision failure, not a scalar calibration failure.

## Decision

Build `conservative-policy-v2` on the same frozen, checksummed train and
validation records while keeping the untouched test labels sealed.

The model uses the same hidden-96, four-head, two-board-block,
one-market-block shared pair encoder and adds two outputs:

1. an auxiliary c90 lower-bound regression head;
2. a groupwise policy logit for every challenger, compared with a fixed
   zero-logit anchor.

Training minimizes:

- balanced groupwise cross-entropy over anchor plus all challengers;
- auxiliary boundary-weighted lower-bound regression at weight 0.25.

Challenger-selected groups receive fixed weight 3.56 and anchor-selected
groups weight 1.0, matching the inverse class ratio measured on the frozen
training split. Inference chooses the highest-logit challenger only when its
logit exceeds the anchor's fixed zero logit. There is no threshold tuning.

Checkpoint selection minimizes validation balanced policy cross-entropy:

`0.5 * mean_anchor_cross_entropy + 0.5 * mean_challenger_cross_entropy`.

The hard balanced policy error remains a diagnostic. The implementation smoke
replaced it as the checkpoint metric before substantive training because its
all-anchor plateau is exactly 0.5 and can trigger patience before logits cross
the fixed decision boundary.

## Frozen Protocol

- Data: the exact v1 train, validation, and untouched test manifests.
- Architecture: hidden 96, four heads, two board blocks, one market block.
- AdamW: learning rate 1e-4, weight decay 1e-4.
- At most 20 epochs, validation patience five.
- No warm start and no test-label access before validation passes.

Validation and untouched test must each satisfy:

- mean teacher policy regret at most 0.20 points;
- exact anchor-or-challenger agreement at least 65%;
- anchor false-positive rate at most 20%;
- exact selected-challenger recall at least 35%;
- lower-bound correlation at least 0.50;
- lower-bound MSE below the zero predictor.

Only then may the same ten-game paired gameplay pilot and runtime/category
gates from ADR 0025 run on seeds 28500-28509. A passing pilot alone may
authorize a disjoint 50-game confirmation.

## Result

The fixed run stopped after nine epochs and 2,304 optimizer steps at
validation patience. Epoch 4 was selected by balanced policy cross-entropy.

| Metric | Best validation | Gate |
|---|---:|---:|
| Balanced policy cross-entropy | 1.892260 | selection metric |
| Mean squared error | 1.498838 | below 4.789431 zero predictor |
| Mean policy regret | 0.089396 | <= 0.20 |
| Exact policy agreement | 0.767188 | >= 0.65 |
| Anchor false-positive rate | 0.000000 | <= 0.20 |
| Selected-challenger recall | 0.000000 | >= 0.35 |
| Lower-bound correlation | 0.669211 | >= 0.50 |

The policy head also collapsed to the anchor. Removing the zero-logit boundary
did not rescue it: the selected challenger was the highest-logit challenger
in only 15.4% of challenger-selected validation groups.

Source inspection found the root target mismatch. R8 sample seeds are derived
from the hidden game seed, while the learner correctly receives only public
afterstates. Exact R8 choices therefore contain deterministic Monte Carlo
noise that is unavailable to the model. The untouched test labels were not
evaluated, no gameplay ran, and no model was promoted.

Further supervised loss changes on these labels are closed. ADR 0027 first
tests whether a higher-sample conservative teacher improves playing strength
and produces a more stable expected-advantage target.
