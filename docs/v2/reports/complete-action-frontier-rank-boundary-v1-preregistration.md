# Complete-Action Frontier Rank Boundary V1 Preregistration

Status: **active preregistered**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-rank-boundary-v1`

The authoritative protocol is
`docs/v2/decisions/0093-frontier-rank-matched-boundary.md`.

One john2 MLX pilot replaces ADR 0092's two smooth extremes with a complete
ordered boundary. Every required target rank is paired with a corresponding
hard nontarget rank and receives its own soft margin loss. Temperature is
`1.0`, margin is `0.5`, and all model, data, selector, width, score-range,
optimizer-scale, and checkpoint-selection contracts remain fixed.

john3 must verify complete target gradient coverage, exactly quota-many hard
nontarget gradients, and a finite maximum-width update. john4 must recover at
least 99% target recall and 90% exact target sets on the 12 widest validation
groups while clipped to ±12. Both audits run independently of the sole john2
training trajectory.

The open pilot must reach 60% train target recall, 5% exact train sets, 50%
validation target recall, 1% exact validation sets, and preserve the existing
winner, confidence, regret, memory, swap, integrity, and sealed-domain gates.
No sweep, duplicate training, new teacher compute, test, gameplay, cloud, or
external compute is authorized.
