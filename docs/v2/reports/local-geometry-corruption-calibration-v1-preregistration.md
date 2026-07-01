# Local Geometry Corruption Calibration V1 Preregistration

Date: 2026-06-16

Experiment ID: `local-geometry-corruption-calibration-v1`

Decision:
[ADR 0123](../decisions/0123-local-geometry-corruption-calibration.md)

## Arms

- 10% deterministic within-query local-geometry corruption;
- 25% deterministic within-query local-geometry corruption; and
- 50% deterministic within-query local-geometry corruption.

## Selection

Choose the smallest rate that removes at least 25% of the extended
train-validation recall gap while damaging validation recall by no more than
0.02 for either the 20-epoch or 200-epoch checkpoint.

This is a nontraining calibration and cannot alter ADR 0120.
