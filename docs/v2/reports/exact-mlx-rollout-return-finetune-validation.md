# Exact MLX Rollout-Return Fine-Tuning Validation

Date: 2026-06-12

Decision: rejected before gameplay under ADR 0065.

## Evidence

| Item | Result |
|---|---:|
| Train trajectory records | 245,603 |
| Train root records | 9,650 |
| Validation trajectory records | 121,246 |
| Validation root records | 4,844 |
| Selected checkpoint | Epoch 7, step 3,360 |
| Training wall time | 189.55 seconds |
| Parent artifact unchanged | Yes |
| Gameplay domain opened | No |

## Held-Out Metrics

| Metric | Parent | Selected | Gate |
|---|---:|---:|---|
| Trajectory RMSE | 5.14326 | 2.99122 | Pass |
| Within-turn residual Pearson | 0.41853 | 0.47761 | Pass |
| Root pairwise accuracy | 0.71834 | 0.71682 | Fail |
| Selected-action top-one | 0.28125 | 0.27500 | Fail |
| Conditional mean regret | 1.05553 | 1.01389 | Pass |

All four personal-turn quartile RMSEs improved. The selected model therefore
learned terminal score-to-go much more accurately, but it did not preserve the
root ordering consumed by search. The failed pairwise and top-one gates close
this exact recipe without gameplay.

## Provenance

- Train manifest:
  `5a041c73c15075e38d3106a77b09b1a33e6597c4d8eee5eea38446490c282ec0`
- Validation manifest:
  `ee98a051423d4448173a8479ddbb2ff7ff614d9b358a82cc95549cca4d33b6e3`
- Final report:
  `7cb3e98a3055869d8f44c4906ddb45ce6a290322fc9eb4b8065265a242fa9943`
- Derived manifest:
  `d2ed8cbbdd5f34ebb1838edc70f45e72f344c3586fc3c4a5bae62e3adbac5db3`
- Derived tensors:
  `1b3ed4a47bc4450129e9f117e822e0af0fe2f7a3de3d932fbcbcf151fdf06c67`

The generated machine-readable report remains at
`artifacts/runs/exact-mlx-rollout-return-v1/final-report.json`.
