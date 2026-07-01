# Exact MLX Joint Return and Root-Ranking Validation

Date: 2026-06-12

Decision: rejected before gameplay under ADR 0066.

## Evidence

| Item | Result |
|---|---:|
| Train trajectory records | 245,603 |
| Train root records | 9,650 |
| Fresh validation trajectory records | 119,753 |
| Fresh validation root records | 4,621 |
| Selected checkpoint | Epoch 12, step 6,720 |
| Training wall time | 231.88 seconds |
| Parent artifact unchanged | Yes |
| Gameplay domain opened | No |

## Held-Out Metrics

| Metric | Parent | Selected | Gate |
|---|---:|---:|---|
| Trajectory RMSE | 5.03061 | 2.88953 | Pass |
| Within-turn residual Pearson | 0.46068 | 0.52172 | Pass |
| Root selection loss | 2.80536 | 2.76335 | Diagnostic pass |
| Root pairwise accuracy | 0.70911 | 0.70573 | Fail |
| Selected-action top-one | 0.27500 | 0.29375 | Pass |
| Conditional mean regret | 1.01325 | 1.28064 | Fail |

All personal-turn quartile RMSEs improved. The root objective also improved
its own selected checkpoint metric and exact selected-action recall. The
remaining errors became more expensive, however, and broad pairwise ordering
degraded. The failed pairwise and regret gates close this exact objective
without gameplay.

## Provenance

- Train manifest:
  `5a041c73c15075e38d3106a77b09b1a33e6597c4d8eee5eea38446490c282ec0`
- Fresh validation manifest:
  `10adbe29432e10a91380c5a43a4391fe9695d7884925d7fe83533c311ced8dcf`
- Final report:
  `240532422ff708dda0bdb12ee3a3dde96e84d3a9ade0bce15e6266aec6e1ac7e`
- Derived manifest:
  `06e125fa3843cbf01c0b20af80dcfe595a189cc3346fd342fe901fba382baf66`
- Derived tensors:
  `fbcce85ba57654a34709c7d305749ea8f4d20e3da9be206c25cfbae69eeda362`

The generated machine-readable report remains at
`artifacts/runs/exact-mlx-joint-return-ranking-v1/final-report.json`.
