# Exact MLX Rollout-Wave Parity

Experiment:
`qualified-legacy-nnue-mlx-exact-rollout-wave-v1-parity-20260612`

## Result

Passed. The exact MLX treatment reproduced the unchanged native qualified
search over the complete 80-decision train trajectory.

| Check | Result |
|---|---:|
| R32 decisions | 80 |
| R32 estimates | 2,494 |
| R32 candidate mismatches | 0 |
| R32 selected-action mismatches | 0 |
| R32 sample-count mismatches | 0 |
| R32 maximum rollout-mean error | 0.0 |
| R600 spot decisions | 3 |
| R600 estimates | 87 |
| R600 mismatches or error | 0 |
| Neural fallbacks | 0 |
| R32 neural batches | 4,030 |
| R32 neural rows | 1,726,630 |
| Native R32 wall time | 44.550 s |
| Exact MLX R32 wall time | 47.783 s |
| MLX/native ratio | 1.073x |

Repeated MLX execution was bit-identical and service shutdown was clean.

## Interpretation

The evaluator-independent rollout refactor is behavior-preserving, and ADR
0058's packed CSR operation is exact enough to preserve strict near ties,
sequential-halving allocation, and the entire stochastic rollout trajectory.
Every treatment neural forward ran through MLX on the Apple GPU.

This qualifies an MLX-backed reproduction of the historical teacher for a
separately frozen gameplay pilot. It is not a model promotion and makes no
claim toward the 100-point target.

Machine-readable report:
`legacy-nnue-v4opp-mlx-exact-rollout-wave-v1.json`

BLAKE3:
`ad17c43f0e55006ca16deb141fbafe3b28c219d98af7848877967dfbe41c75d7`
