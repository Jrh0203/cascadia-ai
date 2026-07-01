# Complete-Action R4800 Identifiability V1 Preregistration

Status: **closed complete; representation or optimization remains material**

Date: 2026-06-16

Experiment ID: `complete-action-r4800-identifiability-v1`

The authoritative protocol is
`docs/v2/decisions/0086-r4800-target-identifiability-audit.md`.

This no-training audit uses only the already-open ADR 0081 train and
validation splits and its frozen selected john2 checkpoint. It measures
finite-sample R4800 winner identifiability, R1200-to-R4800 target stability,
confidence-set coverage, distinguishable-winner recall, and retained regret.

The sealed test, gameplay domains, K2048 screen, new teacher compute, model
changes, and threshold changes are prohibited. Validation is replayed on all
three Macs, and the scientific outputs must be identical.

The audit completed within those bounds. Validation scientific metrics were
identical across john1, john2, and john3. The selected model covered the
R4800 95% confidence set in 86.25% of decisions and recalled 85.53% of
statistically distinguishable winners, so the frozen classification is
`representation_or_optimization_material`. No sealed-test group or gameplay
seed was opened.

Results:

- `docs/v2/reports/complete-action-r4800-identifiability-v1.json`;
- `docs/v2/reports/complete-action-r4800-identifiability-v1.md`;
- `artifacts/experiments/complete-action-r4800-identifiability-v1/manifest.json`.
