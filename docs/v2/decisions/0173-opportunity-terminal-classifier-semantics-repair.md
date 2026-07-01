# ADR 0173: Opportunity Terminal Classifier Semantics Repair

- Status: accepted
- Date: 2026-06-16
- Experiment: `opportunity-cross-attention-mlx-tournament-v1`
- Scope: terminal evidence classification only
- Does not change: data, model topology, initialization, training batches,
  optimization, checkpoints, thresholds, serving measurements, or gameplay
  claims

## Context

The ADR 0166 terminal classifier was audited while all four production arms
were still training and before any final arm report or terminal classification
existed. The audit found two deterministic bookkeeping defects.

First, the classifier accumulated whether any earlier treatment had positive
global and strategic utility, then combined that accumulated state with the
current treatment's protected-slice result. A later treatment with no positive
utility could therefore convert a different treatment's quality-regression
verdict into `opportunity_query_protected_slice_regression`.

Second, ADR 0166 preregistered this lexicographic selection order among fully
eligible treatments:

1. top-64 winner recall;
2. strategic-opportunity recall;
3. top-64 retained regret;
4. R4800 RMSE;
5. complete-decision P99 latency; and
6. peak process RSS.

The implementation omitted R4800 RMSE and peak process RSS from the selector.

Neither defect changes whether an individual arm passes its frozen advancement
gates. They can only change the aggregate failure label or the winner among
multiple already-eligible treatments.

## Decision

Protected-slice attribution is arm-local. A treatment contributes to the
protected-slice failure class only when that same treatment:

- improves global top-64 recall against both controls;
- has the required global bootstrap probability;
- improves strategic-opportunity recall against the parent control;
- has the required strategic bootstrap probability; and
- fails low-supply or independent-draft noninferiority.

The aggregate `utility_positive` flag remains an any-arm reduction used only
for the quality-regression fallback.

The eligible-arm selector now implements all six preregistered criteria in
their exact order, followed only by the frozen arm order as a deterministic
final tie-break.

## Frozen Repair

The repair is distributed as a classifier-only content-addressed bundle. It
imports the unchanged ADR 0172 training bundle for all model, protocol, and
paired-panel code.

Repair bundle:
`artifacts/experiments/opportunity-cross-attention-mlx-tournament-v1/repairs/fa97e459888c8d30842dc25298b624bcc4b43922bb91e1882418db14c79b493c`

Upstream training bundle:
`249819771886a11d235c8f91193bbea8b44143c7da54338522689f99628572dd`

| File | BLAKE3 |
|---|---|
| `tools/opportunity_cross_attention_mlx_report.py` | `2d1e194a6e8212c7a7af4cc093074023add75c21be3898cde1d8746c0872f3fc` |
| `tools/test_opportunity_cross_attention_mlx_report.py` | `7897c831238741ba3eaa8fbfdb92c41750a2870b6e7d016b8d329564320051d7` |

## Verification

Required before the repaired classifier may enter the live queue:

- the complete opportunity classifier and pairwise test suites pass;
- Ruff passes on the repaired classifier and tests;
- a regression proves protected failures cannot move between arms;
- a regression proves RMSE outranks latency and memory;
- a regression proves memory breaks a tie after latency;
- the repair bundle is content-addressed and read-only; and
- the superseded blocked classifier task is administratively cancelled without
  modifying its immutable command.

Qualification result:

- complete opportunity suite: 33 passed;
- Ruff: pass;
- diff check: pass; and
- no final arm report or classification was observed before the bundle was
  frozen.

## Consequences

The four production training arms and their reports remain valid because this
repair changes no scientific input or model computation. The terminal
classification must run from the ADR 0173 repair bundle. The original
classifier remains preserved as audit evidence and must not be executed for the
production verdict.
