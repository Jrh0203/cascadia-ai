# Conditional Tile Optimizer Schedule V1 Preregistration

Date: 2026-06-16

Experiment ID: `conditional-tile-optimizer-schedule-v1`

Decision: [ADR 0120](../decisions/0120-conditional-tile-optimizer-schedule.md)

## Question

Can one frozen late cosine learning-rate decay recover validation
generalization after ADR 0118 proved that 200 epochs at a fixed rate nearly
memorize train while validation regresses?

## Single Treatment

- one from-scratch conditional tile origin;
- exact ADR 0118 model, features, objective, data, seed, batch, AdamW, weight
  decay, width, epoch budget, and train-only checkpoint selection;
- `3e-4` for epochs 1-20;
- one cosine decay from `3e-4` to `3e-6` for epochs 21-200; and
- one validation evaluation after epoch 200.

There is no warm start, early stop, resampling, second seed, schedule sweep, or
validation-directed selection.

## Pass Condition

The treatment must exceed 90% validation tile recall, 98% mixed target recall,
98% mixed winner retention, and every integrated ADR 0115 proposal gate while
also exceeding 95% train recall and passing all pipeline, replay, identity,
resource, and closed-domain gates.

## Mechanical Decision

- Pass: freeze the tile proposal and move to the complete-action selector.
- Fail with a valid pipeline: close this conditional pointwise ranker route;
  do not run another exposure, sampling, or optimizer-schedule variant.
- Invalid pipeline: repair only the failed integrity condition before
  interpreting strength.

The sealed test, gameplay, new teacher compute, cloud, Modal, and external
compute remain closed.
