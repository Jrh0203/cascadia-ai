# Conditional Tile Specialization Attribution V1 Preregistration

Date: 2026-06-16

Experiment ID: `conditional-tile-specialization-attribution-v1`

Decision:
[ADR 0122](../decisions/0122-conditional-tile-specialization-attribution.md)

## Question

Which frozen tile-item feature block acquired the train-only association
measured by ADR 0121?

## Arms

- tile-factor identity `[0, 8)` on john1;
- local geometry `[8, 188)` on john3; and
- descendant summaries `[188, 249)` on john4.

Each arm compares unmodified scoring with one deterministic within-query
cyclic permutation for both the 20-epoch and 200-epoch checkpoints on train
and validation.

## Decision

A block is identified only when its specialization contribution is at least
0.05 and at least 0.02 above the runner-up. Otherwise specialization is
classified as distributed. This audit performs no training and cannot change
ADR 0120.
