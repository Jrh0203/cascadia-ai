# ADR 0052: Exact Hex-Rotation Augmentation

Status: rejected on validation on 2026-06-12. No sealed test or gameplay seed
was opened.

## Context

The full-legal imitation corpus contains 5,120 training positions but
327,680 candidate actions. Board and action coordinates are expressed in a
fixed axial frame, and tile orientation is explicit. Without augmentation,
the ranker must separately learn equivalent placement and wildlife patterns
in six global orientations.

ADR 0050 showed that more action-to-token capacity did not improve
generalization. ADR 0051 showed that a monotonic immediate prior improved
pairwise behavior but did not solve top-rank selection when trained jointly.
The next cheapest root-cause test is exact geometric invariance.

## Decision

Add deterministic training-only rotation augmentation:

1. Sample one rotation in `[0, 5]` for each decision group from the epoch and
   batch seed.
2. Rotate every board's axial coordinates by the same multiple of 60 degrees.
3. Rotate every placed tile orientation by the same amount.
4. Rotate every candidate tile coordinate, wildlife coordinate, and tile
   orientation identically.
5. Preserve market identity, global features, masks, immediate values,
   targets, and candidate ordering.
6. Leave validation and inference unaugmented.

The generic grouped trainer exposes a deterministic augmentation hook. Exact
tests prove one-step coordinate/orientation behavior and six-step round-trip
identity. Resume regenerates the same augmentation sequence from the frozen
seed.

## Frozen Validation Experiment

- Train dataset: `canonical-action-imitation-train-a0155b3613e51112`.
- Validation dataset:
  `canonical-action-imitation-validation-4929d2a8a2bb0a0d`.
- Model: original `shared-state-action-imitation-v1`, hidden 96, four heads,
  two board blocks, one market block.
- Augmentation: one independently sampled exact global rotation per group per
  epoch.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Training: at most 20 epochs, batch 16, patience five, seed 20260615.
- Selection: canonical-orientation validation listwise loss only.
- Command: `make train-imitation-rotations`.

The model advances only if its selected checkpoint reaches:

- validation listwise loss below 2.90;
- top-one accuracy at least 23%;
- top-five recall at least 57%;
- MRR at least 0.40;
- exact augmentation, checkpoint, and dataset integrity.

Passing authorizes a separately registered robustness experiment and fresh
test domain. The rejected ADR 0049 test split remains sealed. Missing any
gate rejects rotation augmentation as a sufficient standalone solution.

## Result

The run completed eleven epochs in 42.4 seconds and stopped after five
non-improving epochs. Epoch six was selected:

- listwise loss 3.011004, missing the 2.90 gate and worse than v1;
- top-one accuracy 20.703%, missing the 23% gate;
- top-five recall 48.047%, missing the 57% gate;
- MRR 0.341837, missing the 0.40 gate;
- pairwise accuracy 87.493%.

Exact rotational invariance did not improve the apprentice and reduced
top-five quality. Orientation sample complexity is not the dominant limit of
the winner-only dataset. The experiment is rejected without test access.
