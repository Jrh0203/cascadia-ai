# Complete-Action R1200 Target Sufficiency V1 Preregistration

Status: **closed rejected before training**

Date: 2026-06-16

Experiment ID: `complete-action-r1200-target-sufficiency-v1`

The authoritative protocol is
`docs/v2/decisions/0087-r1200-cohort-target-sufficiency-audit.md`.

This no-training audit uses only the already-open ADR 0081 train and
validation splits and its frozen selected john2 checkpoint. It tests one
predefined information upper bound: stable R1200-mean ranking within the
qualified K1024 cohort, followed by historical screen order outside that
cohort.

The frozen advance gates require at least 99% R4800 confidence-set coverage,
98% distinguishable-winner recall, 95% exact-winner recall, less than 0.03
retained regret, at least 98% coverage in every phase, and at least 95%
R1200/R4800 confidence-set intersection. Passing authorizes only a
fixed-architecture, set-valued K1024-to-K64 proposer experiment.

The sealed test, gameplay domains, K2048 screen, new teacher compute, model
changes, threshold changes, and external compute are prohibited. Validation
must replay identically on john1, john2, and john3.

The corrected audit completed with identical validation science across all
three Macs. The R1200 cohort oracle reached 97.08% confidence-set coverage,
90.79% distinguishable-winner recall, 95.42% exact-winner recall, and
0.020742 retained regret. It failed the overall coverage,
distinguishable-winner, and every-phase gates, so the proposed
fixed-architecture proposer is not authorized. No training, sealed-test,
gameplay, K2048, or external-compute work opened.

Results:

- `docs/v2/reports/complete-action-r1200-target-sufficiency-v1.json`;
- `docs/v2/reports/complete-action-r1200-target-sufficiency-v1.md`;
- `artifacts/experiments/complete-action-r1200-target-sufficiency-v1/manifest.json`.
