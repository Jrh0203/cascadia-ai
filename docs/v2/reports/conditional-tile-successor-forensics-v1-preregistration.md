# Conditional Tile Successor Forensics V1 Preregistration

Date: 2026-06-16

Experiment ID: `conditional-tile-successor-forensics-v1`

ADR 0119 freezes three no-training, open-data arms while ADR 0118 runs:

| Host | Arm | Decision |
|---|---|---|
| john1 | factor-selector ceiling | fixed factor aggregation versus complete-action selector |
| john3 | sampling mass | target-mass sampling versus optimizer schedule |
| john4 | score scale | cross-stage normalization required or not dominant |

The exact metrics, thresholds, selection order, closed domains, and maximum
compute are defined in
`docs/v2/decisions/0119-conditional-tile-successor-forensics.md`.

The portfolio cannot promote a model, open sealed test or gameplay, alter ADR
0118, or authorize more than its mechanically selected successor mechanism.
