# Determinized Lookahead V2 K16/R4/D4

Status: confirmed positive on 2026-06-10, not promoted.

## Evidence

Candidate recall against K16 was only 89.17% for the promoted K8 policy across
240 decisions. A 10-game pilot then gained 0.425 points.

The disjoint confirmation on seeds 21000-21049 produced:

- K8 mean: 90.810
- K16 mean: 91.555
- paired delta: +0.745
- 95% CI: [0.187, 1.303]
- record: 33-1-16
- paired wall time: 421.778 seconds

K16 improved every wildlife category, Nature Tokens, and net habitat on this
suite. The effect is statistically positive, but its confidence-interval lower
bound missed the pre-registered +0.25 promotion requirement.

K8 therefore remains the interactive policy. K16 is retained as a research
teacher configuration because it is stronger in this confirmation and remains
entirely local and reproducible.
