# ADR 0099 Frontier Supervision Identifiability Preregistration

Experiment: `complete-action-frontier-supervision-identifiability-v1`

Date locked: 2026-06-16

The experiment executes four distinct non-neural audits over the unchanged
open train and validation datasets:

| Host | Audit | Frozen question |
|---|---|---|
| john1 | boundary signal-to-noise | Are nominal top-64 labels statistically separated at the cutoff? |
| john2 | cross-fidelity stability | Do R600 and R1200 independently produce the same learned quota? |
| john3 | 512-draw resampling | Does finite teacher noise reproduce the nominal target set? |
| john4 | expected-rank ceiling | Can uncertainty-aware ordinal supervision retain the R4800 decision signal? |

Every origin report is replayed on the next host in the ring. Scientific
payloads must match bit-for-bit. The sealed test, gameplay, new teacher
compute, neural training, cloud, and external compute remain closed.

Each origin and replay uses exactly eight ordered worker processes. Across the
four simultaneous origins this assigns 32 of the cluster's 40 physical cores
to distinct scientific work while preserving two cores per Mac for system and
cluster services.

Classification and all thresholds are frozen in
`docs/v2/decisions/0099-frontier-supervision-identifiability.md`. No threshold,
draw count, seed, scoring rule, or target definition may change after any
result is observed.

Pre-result correction: the first john2 cross-fidelity process stopped before
producing metrics because some R600 cohorts could not fill width 64. The
corrected frozen gate requires 100% fillable-group coverage and computes the
original 80% recall and 25% exact-set tests conditionally on fillable groups.
All first-launch outputs are quarantined and all four origins rerun under one
corrected source digest.
