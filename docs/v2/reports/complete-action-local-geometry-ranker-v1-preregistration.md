# Complete-Action Local-Geometry Ranker V1 Preregistration

Status: **completed; rejected on validation**

Date: 2026-06-16

Experiment ID: `complete-action-local-geometry-ranker-v1`

The authoritative protocol is
`docs/v2/decisions/0088-complete-action-local-geometry-ranker.md`.

This experiment changes one observable representation mechanism. It adds a
rotation-canonical candidate-to-active-board path for the six tile neighbors,
wildlife target tile, and six wildlife neighbors. The complete dataset,
teacher labels, historical screen priors, loss, optimizer, paired seeds,
training budget, augmentation, checkpoint selection, and sealed domains stay
fixed.

The treatment must pass exact geometry, six-rotation invariance,
zero-initialization parity, maximum-width, finite-gradient, identity, and
sealed-boundary tests before three replicas run concurrently on john1, john2,
and john3.

The selected replica must exceed 98% exact top-64 R4800-winner recall, reach
at least 99% R4800 confidence-set coverage and 98% distinguishable-winner
recall, retain less than 0.15 mean R4800 regret, satisfy all phase/subset
gates, and remain inside the existing throughput, latency, memory, and swap
envelope. Passing authorizes only a separately frozen sealed test.

New teacher compute, a fourth replica, architecture or optimizer sweeps,
threshold changes, sealed-test access, gameplay, K2048, and external compute
are prohibited.

The completed result is recorded in
`docs/v2/reports/complete-action-local-geometry-ranker-v1-rejection.md`.
The selected john2 replica reached 74.17% exact recall, 87.92% confidence-set
coverage, 88.16% distinguishable-winner recall, and 0.093757 retained regret.
It passed integrity and performance but failed the frozen winner-recovery,
phase, and subset gates. Sealed test and gameplay stayed closed unopened.
