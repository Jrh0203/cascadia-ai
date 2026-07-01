# Oracle-Proposal Complete-Action Selector V1 Preregistration

Date: 2026-06-16

Experiment ID: `oracle-proposal-complete-action-selector-v1`

ADR 0126 reuses the four exact ADR 0097 pre-compression action-factor
architectures after filtering the open complete-action cache to the oracle
hierarchical `16 / 32 / 8` proposal. This isolates whether candidate-set width
caused the earlier factor-integration failure.

The train and validation filters are disjoint, action-hash aligned, and
checksum-manifested. They preserve phase, Nature Token availability, and
independent-draft winner metadata so every frozen slice gate is mechanically
enforced. Each model trains for 20 epochs from its original seed with
train-only checkpoint selection. Validation is opened once after selection.

The feasibility gate requires at least 95% train target recall, 50% train
exact sets, 90% validation target recall, 98% validation winner retention,
less than 0.15 validation regret, and every ADR 0115 phase/subset guardrail.

This open-data experiment does not alter ADR 0120, authorize a tile result,
promote a selector, or open sealed gameplay.
