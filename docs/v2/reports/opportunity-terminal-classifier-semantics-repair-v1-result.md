# Opportunity Terminal Classifier Semantics Repair V1

Date: 2026-06-16

Decision: ADR 0173

Experiment: `opportunity-cross-attention-mlx-tournament-v1`

## Result

The terminal classifier is repaired and frozen before any final production arm
report or terminal outcome existed. The four ADR 0172 production training runs
remain valid and continue unchanged.

Two deterministic bookkeeping defects were corrected:

1. protected-slice failure attribution now belongs only to the same treatment
   that first establishes positive global and strategic utility; and
2. eligible-arm selection now implements the full preregistered order:
   top-64 recall, strategic recall, retained regret, R4800 RMSE, P99 latency,
   and peak process RSS.

No metric, threshold, bootstrap seed, model tensor, training batch, checkpoint,
serving measurement, or gameplay claim changed.

## Frozen Artifact

Classifier-only repair bundle:

`artifacts/experiments/opportunity-cross-attention-mlx-tournament-v1/repairs/fa97e459888c8d30842dc25298b624bcc4b43922bb91e1882418db14c79b493c`

Upstream ADR 0172 training bundle:

`249819771886a11d235c8f91193bbea8b44143c7da54338522689f99628572dd`

| File | BLAKE3 |
|---|---|
| `tools/opportunity_cross_attention_mlx_report.py` | `2d1e194a6e8212c7a7af4cc093074023add75c21be3898cde1d8746c0872f3fc` |
| `tools/test_opportunity_cross_attention_mlx_report.py` | `7897c831238741ba3eaa8fbfdb92c41750a2870b6e7d016b8d329564320051d7` |

The canonical `bundle_identity` hashes to the directory name. The bundle and
its files are read-only.

## Verification

- Complete opportunity suite: 33 passed.
- Isolated repair-bundle classifier tests against the immutable upstream
  Python source: 4 passed.
- Ruff: pass.
- Git diff check: pass.
- Queue validation: pass.
- Experiment-ledger validation: pass.
- Dashboard API reflects the repaired classifier stage.

The regression suite includes:

- an arm-order case that previously misclassified an unrelated protected-slice
  failure;
- a tie where RMSE must outrank lower latency and memory; and
- a tie where RSS must decide only after latency.

## Queue Transition

The original blocked task `oppquery-v3-classify` was administratively
cancelled with its immutable command and audit record preserved.

The replacement task is:

`oppquery-v3-classify-adr0173`

It depends on the unchanged production collection and untouched-C0 tasks,
imports all scientific model and paired-panel code from the ADR 0172 bundle,
and executes only the content-addressed ADR 0173 classifier.

## Claim Boundary

This repair establishes classifier correctness, not model quality. It does not
authorize gameplay qualification, change the champion, or establish progress
toward the 100-point target. Those claims remain gated on the four completed
arm reports and the repaired terminal classification.
