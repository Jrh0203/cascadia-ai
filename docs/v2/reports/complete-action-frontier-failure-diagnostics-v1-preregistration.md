# Complete-Action Frontier Failure Diagnostics V1 Preregistration

Status: **active preregistered**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-failure-diagnostics-v1`

The authoritative protocol is
`docs/v2/decisions/0090-frontier-ranker-failure-classification.md`.

Four independent open-data diagnostics run concurrently: train fit on john1,
exact observable collision on john2, objective gradients on john3, and error
anatomy on john4. They share ADR 0089's selected checkpoint and immutable open
train/validation corpus but do not duplicate work or train a model.

Frozen thresholds classify underfit, generalization, exact contradictory
observations, auxiliary gradient domination/conflict, and concentrated error
slices. A deterministic combine step selects one mechanism and authorizes only
one single-host MLX pilot family. The other three Macs remain available for
independent work; training replication requires a later explicit validation
question.

Sealed test, gameplay, new teacher compute, wider brute force, threshold
changes, duplicate diagnostics, cloud, and external compute are prohibited.
