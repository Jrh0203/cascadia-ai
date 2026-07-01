# Complete-Action Graded Oracle Ranker V1 Preregistration

Status: **closed rejected on validation; sealed test and gameplay unopened**

Date: 2026-06-15

Experiment ID: `complete-action-graded-oracle-ranker-v1`

The authoritative protocol is frozen in:

- `docs/v2/decisions/0081-complete-action-graded-oracle-ranker.md`;
- `docs/v2/decisions/0082-complete-action-graded-oracle-ranker-test.md`;
- `docs/v2/decisions/0083-complete-action-graded-oracle-gameplay.md`;
- `docs/v2/decisions/0084-graded-oracle-pretraining-contract-correction.md`;
- `docs/v2/decisions/0085-graded-oracle-observable-input-correction.md`.

ADR 0084 was locked after lossless conversion exposed parser-feature and
indivisible-group-width contradictions, but before model initialization,
training, validation evaluation, or any Python access to sealed-test groups.

ADR 0085 was locked after the first one-epoch implementation diagnostic
revealed that source and fidelity allocation bits had accidentally entered
the model prior. Those three runs were terminated and disqualified before
model selection. Their evidence is preserved, their checkpoints cannot load
under the corrected schema, and the sealed test remained unopened.

This experiment converts the completed K1024 full-legal corpus into a
lossless grouped dataset, trains three fixed MLX residual rankers across
john1, john2, and john3, requires greater than 98% top-64 R4800-winner recall
and less than 0.15 retained regret on game-disjoint validation and sealed test
games, then conditionally runs one preregistered learned-screen gameplay
pilot.

No gameplay seed, test metric, alternate architecture, hyperparameter sweep,
warm start, K2048 run, or external compute is authorized outside those ADRs.

The corrected three-replica run completed on 2026-06-16. The john2 replica
won the frozen selection objective at 0.090184 retained mean R4800 regret,
but reached only 73.33% top-64 R4800-winner recall. Every overall, phase, and
subset recall gate failed. Cross-host metrics were bit-identical, and the
selected model exceeded the performance gates on all three Macs. ADR 0082
therefore closed unopened, ADR 0083 closed unopened, and no threshold was
changed after observing validation.

Final evidence:
`docs/v2/reports/complete-action-graded-oracle-ranker-v1-rejection.md`.
