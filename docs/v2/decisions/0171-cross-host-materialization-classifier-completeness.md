# ADR 0171: Cross-Host Materialization Classifier Completeness

- Status: accepted
- Date: 2026-06-17
- Experiment: `exact-r2-preverified-vectorized-materialization-v1`
- Scope: terminal qualification completeness
- Scientific measurements affected: none

## Context

Before any queued four-host qualification task started, review found that the
ADR 0167 classifier accepted two complete crossed validation reports and one
prediction report but did not require:

- the preregistered complete train-split parity report;
- exact source identity equality across all reports;
- one shared open-data proof; or
- execution on four distinct hosts.

The preregistration already names
`exact_r2_vectorized_materialization_cross_host_inconsistent` as a terminal
classification. The implementation did not yet emit it. Leaving that gap would
allow a structurally incomplete or source-drifted campaign to promote.

## Decision

Qualification now requires exactly:

1. two complete validation reports, one `legacy-first` and one
   `vectorized-first`;
2. one complete train report;
3. one complete frozen-C0 prediction report;
4. the frozen action and decision counts for each split;
5. exact equality to the classifier's immutable source identity;
6. one identical open-data proof ID across all four reports; and
7. four distinct runtime host identities.

If the shape is complete but any source, data-proof, or host identity differs,
the classifier returns
`exact_r2_vectorized_materialization_cross_host_inconsistent`.

Structural completeness is checked before cross-host consistency. Feature and
prediction parity, memory, swap, and performance gates remain unchanged.

## Queue Consequence

The first six qualification tasks were installed but remained blocked and
never started. They are administratively superseded before execution. A new
content-addressed bundle and task graph carry the ADR 0171 source hash.

No measurement from the earlier complete john1 run is promoted through the
new classifier; all terminal evidence is regenerated from the superseding
immutable bundle.
