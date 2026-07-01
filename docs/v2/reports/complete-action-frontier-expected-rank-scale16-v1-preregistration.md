# Complete-Action Frontier Expected-Rank Scale 16 V1 Preregistration

Status: frozen before treatment metrics.

Decision:
`docs/v2/decisions/0101-frontier-expected-rank-scale16.md`

Experiment:
`complete-action-frontier-expected-rank-scale16-v1`

This experiment changes one value only: expected-rank target scale 64 becomes
16. The expected ranks, model, initialization seed, optimizer, datasets,
frontier anchors, proposal width, residual range, student temperature,
checkpoint selection, quality gates, and sealed boundaries remain unchanged.

The scale was selected before treatment training because the complete train
audit places 93.76% of target mass inside the deployed nonfrontier set at
scale 16, versus 44.84% at scale 64. Open validation independently mirrors
93.75%; it did not choose the scale.

Cluster allocation is asymmetric and throughput-oriented:

- john2 builds canonical caches and trains the only model;
- john1 independently rebuilds both caches, audits target/gradient alignment,
  and later replays the selected checkpoint;
- john3 runs 32-step optimization audits on the 12 widest train groups; and
- john4 measures complete baseline, subset, and bounded-reachability anatomy.

Promotion requires the unchanged ADR 0100 train-fit, open-validation R4800,
phase, subset, replay, performance, memory, and integrity gates. Failure stops
without a second seed, warm start, target-scale variant, or sealed evaluation.

