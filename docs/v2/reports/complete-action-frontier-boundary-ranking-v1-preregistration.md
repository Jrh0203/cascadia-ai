# Complete-Action Frontier Boundary Ranking V1 Preregistration

Status: **active preregistered**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-boundary-ranking-v1`

The authoritative protocol is
`docs/v2/decisions/0092-frontier-topk-boundary-ranking.md`.

One john2 MLX pilot warm-starts the selected ADR 0089 checkpoint and changes
only the optimization surrogate. A conservative smooth top-K boundary loss
pushes the weakest required target above the strongest eligible nontarget with
temperature `0.25` and margin `0.5`. Architecture, observable features,
datasets, ±12 residual range, selector, proposal width, and augmentation are
unchanged.

Before training, john3 must pass a real maximum-width gradient and optimizer
step. In parallel, john4 must show that direct bounded score-space optimization
recovers at least 99% of target slots and 90% of exact target sets on the 12
widest validation decisions. john1 coordinates identities and evaluates the
selected checkpoint.

The pilot gates are the same as ADR 0091 for direct comparability: 60% train
target recall, 5% exact train sets, 50% validation target recall, 1% exact
validation sets, 75% winner recall, 90% confidence coverage, regret below
0.15, complete finite scoring, RSS below 4 GiB, zero swaps, and no sealed-test
access.

A second seed, duplicate training, parameter sweep, architecture change, new
teacher compute, sealed test, gameplay, cloud, and external compute are
prohibited unless the pilot passes.
