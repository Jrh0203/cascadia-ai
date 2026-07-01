# Conditional Tile Target-Only Objective V1 Preregistration

Date: 2026-06-16

Experiment ID: `conditional-tile-target-only-objective-v1`

## Question

Does removing the rank-regression and listwise terms allow the unchanged
conditional tile model to learn the actual top-32 membership target closely
enough to restore the hierarchical proposal?

## Frozen Treatment

- Reuse the immutable ADR 0115 train and validation factor caches.
- Reuse the exact tile model inputs, architecture, hidden width 256, and width
  32.
- Train from scratch for 20 epochs with seed `2026061648`, batch size 32,
  AdamW `3e-4`, and weight decay `1e-4`.
- Optimize only the existing class-balanced top-width membership BCE.
- Select the checkpoint on train recall and exact-query recovery.
- Evaluate validation once after selection.

No other model, feature, width, optimizer, schedule, seed, or objective change
is permitted.

## Decision Rule

The treatment must exceed 95% train and 90% validation tile factor recall,
then exceed 98% validation target-action recall and 98% validation R4800 winner
retention with every other stage oracle-perfect. Its fully learned hierarchy
must also pass every ADR 0115 proposal gate.

Pass as `target_only_tile_objective_sufficient`; otherwise reject as
`target_only_tile_objective_insufficient`.

## Compute

Run one origin on one Mac, one required cross-host replay, one mixed-stage
ceiling, and one integrated evaluation. The remaining Macs continue distinct
closeout, correctness, reporting, or independent research work. No duplicate
discovery seed is authorized.
