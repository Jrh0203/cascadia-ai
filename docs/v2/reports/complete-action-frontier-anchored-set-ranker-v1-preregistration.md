# Complete-Action Frontier-Anchored Set Ranker V1 Preregistration

Status: **active preregistered**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-anchored-set-ranker-v1`

The authoritative protocol is
`docs/v2/decisions/0089-complete-action-frontier-anchored-set-ranker.md`.

This experiment replaces scalar all-action ranking with an exact width-64
set-retention mechanism. Every deterministic champion/frontier action is
anchored in the proposal. The unchanged observable MLX ranker fills only the
remaining nonfrontier slots and is trained against the R1200 nonfrontier set,
with frozen listwise and screen-only regularization terms.

Before training, the deterministic target ceiling must independently pass on
john1, john2, john3, and john4 with bit-identical scientific outputs. It must
exceed 98% exact R4800-winner recall, reach at least 99% confidence-set
coverage and 98% distinguishable-winner recall, retain less than 0.03 mean
R4800 regret, clear every phase and eligible subset gate, score every action
once at width 64, and leave the sealed test unopened.

Only a passing ceiling authorizes correctness smokes and four paired MLX
replicas using seeds `2026061601` through `2026061604`. The selected model
must then pass the complete validation, phase, subset, portability,
throughput, latency, memory, and swap contract before a separate sealed-test
ADR may be written.

New teacher compute, wider screening, architecture/loss/optimizer sweeps,
warm starts, a fifth seed, sealed-test access, gameplay, K2048, and external
compute are prohibited.
