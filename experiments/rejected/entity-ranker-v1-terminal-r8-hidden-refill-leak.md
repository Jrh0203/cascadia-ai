# Entity Ranker V1 Terminal R8: Rejected Before Training

The preregistered terminal R8 distillation experiment was stopped during
validation collection because its candidate feature boundary leaked the actual
hidden post-draft refill.

Completed artifacts:

- train: 64 games, 5,120 groups, 76,741 candidates, manifest BLAKE3
  `9d2b2a4a757f9815f96b650a731c9aa70d802664ee74bc780e31c86782bb85a1`;
- validation: nine of 16 games, 720 groups, 10,960 candidates, manifest BLAKE3
  `edd43a29fff2e35346a7d1f64b51fd63e64c76071866ffbf85483a7d8a023629`.

The teacher values were fair R8 means over public-information
redeterminizations. The nested afterstate record instead called the complete
transition and encoded the real hidden refill. This was both target-feature
mismatch and information unavailable to a legal policy.

No training, promotion, or gameplay evaluation occurred. The artifacts remain
preserved under `artifacts/datasets/rejected/` for forensic reproducibility and
must never be used for model fitting.
