# Conditional Tile Extended Exposure V1 Preregistration

Date: 2026-06-16

Experiment ID: `conditional-tile-extended-exposure-v1`

## Question

Does increasing the unchanged full-cache target-only tile ranker from 20 to
200 epochs close the learned hierarchical proposal gap?

## Frozen Treatment

Reuse ADR 0116 exactly: architecture, inputs, width 32, balanced membership
BCE, AdamW `3e-4`, weight decay `1e-4`, batch 32, seed `2026061648`, caches,
from-scratch initialization, and train-only checkpoint selection. Change only
the total epoch budget from 20 to 200.

## Decision Rule

The selected checkpoint must exceed 95% train and 90% validation tile recall,
then exceed 98% validation target recall and winner retention with every other
stage oracle-perfect. The frozen integrated proposal must also pass every ADR
0115 proposal gate.

Pass as `extended_exposure_tile_sufficient`; otherwise reject as
`extended_exposure_tile_insufficient`. Pipeline invalidity has precedence.

## Compute

john2 trains the sole origin. john3 replays it, john4 measures the mixed
ceiling, and john1 runs integration and reporting. No replica, sweep, early
stop, new data, sealed test, gameplay, cloud, or external compute is
authorized.
