# ADR 0184: Exclude Host-Local Paths From O1 Scientific Identity

**Status:** Accepted  
**Date:** 2026-06-17  
**Repairs:** O1 reuse-audit invalid launch 2

## Context

The corrected `o1reuse-v2` launch replayed the complete frozen corpus on john4
and john2:

- 80 games;
- 6,400 positions;
- 409,600 candidate actions;
- exact position, action, transition, terminal, and split-isolation checks.

Both executions produced identical scientific measurements. Mechanical
classification still failed because each serialized `DatasetAudit` contained
its absolute host-local `root`. The root was included in both the compared
dataset object and `scientific_blake3`, so `/Users/john4/...` and
`/Users/john2/...` necessarily yielded different scientific identities.

An input locator is execution provenance. Dataset identity is already bound by
dataset ID, split, manifest BLAKE3, teacher strategy, teacher weight BLAKE3,
record totals, and the complete exact audit.

## Decision

1. Remove `root` from `DatasetAudit`.
2. Record the two supplied paths as `provenance.dataset_roots`.
3. Increment the audit report schema from 1 to 2.
4. Continue comparing complete scientific dataset, overlap, recoverability,
   claim-boundary, and digest fields across hosts.
5. Require exactly two nonempty host-local dataset roots in execution
   provenance, but never compare or hash them as scientific content.
6. Namespace every launch artifact under
   `launches/<immutable-bundle-id>/` so retries cannot overwrite prior launch
   evidence.
7. Publish the accepted classification to both its immutable launch directory
   and the campaign's canonical `classification.json`.

## Relaunch Rules

- Preserve all `o1reuse-v2` reports and the failed classifier attempt.
- Treat launch 2 as infrastructure-invalid at closeout, not as a negative
  scientific result.
- Rebuild the executable and immutable source bundle.
- Relaunch with `o1reuse-v3` task IDs and bundle-namespaced artifacts.
- Keep the original hypotheses, frozen datasets, and success gates unchanged.

## Consequences

Cross-host replay now means identical scientific content rather than identical
filesystem layout. The same launch-addressing rule also gives later campaign
repairs immutable, inspectable provenance.
