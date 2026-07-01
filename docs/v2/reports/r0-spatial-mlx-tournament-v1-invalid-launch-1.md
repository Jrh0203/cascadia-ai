# R0 Spatial MLX Tournament V1 Invalid Launch 1

Date: 2026-06-17

Experiment: `r0-spatial-mlx-tournament-v1`

Status: invalidated before queue application; zero optimizer steps

## What Happened

The first sealed-bundle preflight was executed locally before the production
queue was applied. Python imported the campaign modules without disabling
bytecode writes and created `__pycache__` files inside the source bundle.
`validate_bundle` then detected those unmanifested files and failed closed.

The affected, unlaunched identities were:

- bundle:
  `2af77e4b8d2a9d60cded05e82fb68babc528b662f1c2fc9e112e9b1831ce8b0d`;
- authorization:
  `6aef5e13081be8bdd527f3301e0301676d1fc509994bf2615de0cf36801718b5`;
- inert task specification:
  `402a2cd3b5cd57a0c44c5cb39e58678200cc31976686579c1365555a79a182e4`.

No task was added to the live queue and no cache export, optimizer step,
evaluation, or gameplay run began under these identities.

## Root Cause

The queue invoked the virtual-environment Python interpreter directly from the
content-addressed bundle. Python's default bytecode-cache behavior writes
beside imported modules. The bundle validator correctly treats any extra file
as identity drift.

## Permanent Repair

Two independent protections were added:

1. every frozen-bundle Python command now passes `-B`, including preflight,
   training, collection, classification, and the portable historical-441
   command; and
2. `rust_experiment_bundle.py` now removes write permission from every file
   and directory after validating and installing a bundle. Reusing an existing
   valid bundle also reapplies the permission seal.

Regression tests assert both the `-B` command contract and read-only bundle
permissions. The focused R0 and bundle suite passes after the repair.

## Disposition

The contaminated bundle, its authorization, and its inert queue specification
are retained under the experiment's `invalidated/` directory. Production must
use a newly built bundle, newly issued authorization, and newly reviewed queue
specification.
