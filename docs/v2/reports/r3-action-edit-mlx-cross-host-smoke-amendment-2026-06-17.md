# R3 Action-Edit MLX Cross-Host Smoke Amendment

Date: 2026-06-17  
Experiment: `r3-action-edit-mlx-comparison-v1`  
Protocol: `r3-action-edit-mlx-matched-comparison-v1`  
ADR: `0150`

## Status

**PASS: numerical parity established before production.**

No production arm, gameplay benchmark, sealed-test evaluation, or model
promotion had started when this amendment was frozen.

## Why The Original Check Was Invalid

ADR 0150 originally required the john1 and john4 10-step smoke loss trace,
final checkpoint, and fixed predictions to be byte-identical.

Both hosts are 10-core, 16 GB Apple M4 Mac minis with MLX `0.31.2`, Python
`3.12.13`, and the MLX GPU device. The bounded runs used the same:

- radius-1 arm;
- cache ID
  `25f5dfdc4d691987419f59dd107c7577a27c707a597050793aa2d656255cbb83`;
- exact S1 cache ID
  `2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15`;
- 10 ordered scientific batches;
- candidate counts;
- initialization tensor hash; and
- model parameter graph.

The john1/john4 loss traces differed only at step 10 by
`1.9073486328125e-06`. Repeating the run on john1 produced the same maximum
loss drift, proving this was within-host MLX GPU nondeterminism rather than
host, data, source, or transfer drift.

## Frozen Numerical-Parity Gates

Exact:

- scientific batch identities;
- per-step candidate counts;
- initial parameter tensor identity;
- parameter count and layout;
- prediction-panel action identities; and
- stable prediction-panel ranking.

Numerical:

| Quantity | Maximum allowed |
|---|---:|
| Loss max absolute drift | `1e-4` |
| Loss max relative drift | `1e-5` |
| Parameter max absolute drift | `1e-4` |
| Parameter mean absolute drift | `1e-6` |
| Prediction score max absolute drift | `1e-4` |
| Prediction uncertainty max absolute drift | `1e-5` |

These tolerances are at least 27 times larger than the largest observed
low-order drift, while remaining 500 times smaller than the experiment's
smallest `0.05` value-quality noninferiority margin.

## First Cross-Host Proof

Comparator:
`tools/r3_action_edit_mlx_smoke_compare.py`

Proof artifact:
`artifacts/experiments/r3-action-edit-mlx-comparison-v1/smoke-runs/cross-host-parity-v1.json`

Proof ID:
`fe189359d217d594ec1c241c14ddc43902066533165c2a9ffb03cf15c39dc481`

Observed:

| Quantity | Measured |
|---|---:|
| Loss max absolute drift | `1.9073486328125e-06` |
| Loss max relative drift | `7.375306794324377e-08` |
| Changed checkpoint scalars | `225,283 / 552,770` |
| Parameter max absolute drift | `3.689900040626526e-06` |
| Parameter mean absolute drift | `6.071621787337557e-09` |
| Prediction score max absolute drift | `0.0` |
| Prediction uncertainty max absolute drift | `1.1920928955078125e-07` |

All exact identities and all numerical gates passed. Both reports and both
complete model checkpoints are checksum-bound into the proof.

## Decision

Use numerical parity, not impossible bitwise floating-point equality, for the
pre-production replay. Keep classification-file order invariance byte-exact;
that path is deterministic JSON and remains an exact requirement.
