# Complete-Action Frontier Expected-Rank V1 Preregistration

Status: frozen before treatment metrics.

Decision: `docs/v2/decisions/0100-frontier-expected-rank-supervision.md`

Experiment: `complete-action-frontier-expected-rank-v1`

This experiment tests one change only: replace unstable hard R1200 top-64
membership with the continuous uncertainty-aware expected-rank distribution
authorized by ADR 0099.

The unchanged ranker starts from scratch on john2 with seed `2026061626`.
Expected ranks are converted to width-scaled exponential mass with denominator
64, and temperature-2 student cross entropy is optimized over every
nonfrontier action. The architecture, observable features, screen prior,
bounded residual, selector, width, datasets, optimizer, and validation
definitions remain frozen.

Cluster allocation is intentionally asymmetric:

- john2 builds the canonical cache and trains the only model;
- john1 independently reproduces both caches, then replays the selected model;
- john3 runs the fixed 32-step widest-group gradient audit; and
- john4 measures screen-baseline and generalization-error anatomy.

Promotion requires material train fit, open-validation transfer, the full
R4800 width-64 quality gates, deterministic replay, performance and memory
gates, and sealed-boundary integrity. A failed pilot stops without a second
seed or treatment.
