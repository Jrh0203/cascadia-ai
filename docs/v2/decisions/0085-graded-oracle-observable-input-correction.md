# ADR 0085: Graded-Oracle Observable Input Correction

Status: locked after an invalid one-epoch implementation diagnostic and
before any valid model training, model selection, sealed-test access, or
gameplay evaluation.

Date: 2026-06-16

Experiment ID: `complete-action-graded-oracle-ranker-v1`

## Context

ADR 0081 prohibits hidden-state features and defines an observable ranker.
The grouped dataset correctly stores `source_flags` and `fidelity_mask` for
target construction, integrity checks, and diagnostics. The first Python
decoder implementation nevertheless appended ten source bits and three
fidelity bits to the model's screen-prior vector.

Those bits are teacher allocation provenance, not live game observations.
They reveal which actions received R1200 and R4800 evaluation. Because the
R4800 winner is selected from that deeper-evaluated set, the model could learn
where the answer was measured instead of learning action quality.

The defect was detected after one epoch on each preregistered host. Initial
validation was identical at 71.667% top-64 recall and 0.1130 retained regret.
After one invalid epoch, john1 and john3 reported 100% recall and zero regret;
john2 reported 98.333% recall and 0.0056 regret. This discontinuity was an
implementation alarm, not a research result.

All three processes were terminated with exit code 143. No invalid checkpoint
was selected, promoted, tested, or used in gameplay. The sealed-test dataset
was not opened by Python or any model.

## Decision

1. The model prior is exactly `observable-screen-priors-v1`, with eight
   live-computable values in this order:
   - historical model immediate score divided by 100;
   - historical model remaining value divided by 100;
   - historical screen value divided by 100;
   - screen rank divided by 4096;
   - inverse screen rank;
   - uniform market-survival proxy;
   - visible wildlife count divided by 4;
   - public-bag wildlife count divided by 20.
2. `source_flags` and `fidelity_mask` remain in the decoded batch solely for
   supervision, integrity checks, and diagnostics. They are never passed to
   the model.
3. A regression test mutates every provenance bit and requires byte-identical
   model prior features.
4. The serialized graded-oracle model configuration advances from schema 1
   to schema 2 and records `prior_feature_schema`. Schema-1 checkpoints are
   rejected before weights are loaded.
5. The three invalid runs are disqualified implementation diagnostics. Their
   metadata, metrics, checkpoint manifests, cursor states, source snapshots,
   termination events, and full file hashes are preserved under
   `artifacts/experiments/complete-action-graded-oracle-ranker-v1/invalid-runs-teacher-provenance-leakage/`.
   Their large model and optimizer files are deleted after hashing and may
   never be resumed or promoted.
6. The valid experiment restarts from fresh initialization with the same
   frozen datasets, three seeds, architecture, loss, optimizer, epoch budget,
   validation gates, and host assignments. These are replacement executions
   of a corrected implementation, not additional statistical replicas.

## Consequences

The prior projection input width changes from the invalid 21 values to the
intended eight observable values. This is not a hyperparameter choice or
post-validation adjustment; it restores the model boundary frozen by ADR
0081. Training may restart only after focused tests, a real maximum-width
forward/backward smoke, and cross-host source-hash equality all pass.
