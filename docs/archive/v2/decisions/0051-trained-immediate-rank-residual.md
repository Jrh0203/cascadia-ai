# ADR 0051: Trained Immediate-Rank Residual

Status: rejected on validation on 2026-06-12. No sealed test or gameplay seed
was opened.

## Context

ADR 0049's learned ranker substantially improved over exact immediate ordering
when the teacher selected strategically delayed actions, but often displaced
the teacher choice when it was already immediate-rank one. ADR 0050 showed
that adding action-query cross-attention did not fix the problem.

A group-normalized reciprocal-immediate-rank prior complements the learned
v1 score. Fixed coefficient 0.08 improved validation from 19.453% to 24.063%
top-one, 50.547% to 58.203% top-five, and 0.347332 to 0.401136 MRR.
Leave-one-game-out coefficient selection chose 0.08 in 13 of 16 folds and
retained 23.359% top-one, 58.047% top-five, and 0.396801 MRR. The signal is
stable enough to train around, but not strong enough for a post-hoc fresh-test
attempt.

## Decision

Create `shared-state-action-residual-ranker-v2`:

1. Keep the proven v1 shared-state/action architecture.
2. Standardize neural logits within each legal-action group.
3. Compute a masked, group-standardized reciprocal immediate-rank prior.
4. Add the prior at frozen coefficient 0.08.
5. Train the neural branch end to end against the combined logits, forcing it
   to learn strategic corrections rather than rediscover a monotonic
   short-horizon ordering.
6. Persist architecture and coefficient in every checkpoint and promoted
   model manifest. Training, evaluation, and service use the same scoring
   function.

The model still scores every canonical legal action in one MLX request.

## Frozen Validation Experiment

- Train dataset: `canonical-action-imitation-train-a0155b3613e51112`.
- Validation dataset:
  `canonical-action-imitation-validation-4929d2a8a2bb0a0d`.
- Model: pooled shared-state ranker, hidden 96, four heads, two board blocks,
  one market block, feed-forward multiplier three.
- Immediate-rank prior: reciprocal rank, masked group standardization,
  coefficient 0.08.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Training: at most 20 epochs, batch 16 groups, patience five, seed 20260614.
- Selection: validation listwise loss only.
- Command: `make train-imitation-residual`.

The model advances only if its selected checkpoint reaches:

- validation listwise loss below 2.90;
- top-one accuracy at least 24%;
- top-five recall at least 58.5%;
- MRR at least 0.405;
- exact checkpoint, scorer, and dataset integrity.

Passing authorizes only a separately registered robustness check and fresh
test domain. The ADR 0049 test split remains sealed. Missing any gate rejects
the trained-residual formulation at coefficient 0.08.

## Result

The run completed eleven epochs in 41.5 seconds and stopped after five
non-improving epochs. Epoch six was selected by listwise loss:

- listwise loss 2.960613, missing the 2.90 gate;
- top-one accuracy 21.328%, missing the 24% gate;
- top-five recall 54.844%, missing the 58.5% gate;
- MRR 0.371453, missing the 0.405 gate;
- pairwise accuracy 89.866%.

Joint training improved pairwise ordering over v1 but learned around the
fixed prior instead of retaining the post-hoc top-rank gain. The formulation
missed every advancement gate and is rejected without test access.
