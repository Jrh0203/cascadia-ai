# Conditional Tile Generalization Forensics V1 Preregistration

Date: 2026-06-16

Experiment ID: `conditional-tile-generalization-forensics-v1`

Decision:
[ADR 0121](../decisions/0121-conditional-tile-generalization-forensics.md)

## Independent Questions

1. Do exact deployed pointwise observables carry contradictory target labels?
2. Is validation materially outside the train input distribution?
3. Did fixed-rate late training expand normalized margins only on train?

## Frozen Thresholds

- Aliasing is material at 1% affected positive mass or 1% contradictory
  cross-split overlap.
- Covariate shift is material at width JSD 0.10, 10% of active dimensions
  above absolute SMD 0.50, or 1% validation cells outside train support.
- Margin specialization requires at least +0.50 train median improvement, at
  most +0.10 validation improvement, and at least +0.50 expansion of the
  train-validation gap.

All three arms use only open caches and frozen checkpoints. They perform no
training and cannot change ADR 0120's preregistered treatment or gates.
