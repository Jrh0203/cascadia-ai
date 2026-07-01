# ADR 0178: V1 Score Anatomy Role-Neutral Checkpoint Parity

- Status: accepted
- Date: 2026-06-16
- Experiment: `v1-score-anatomy-matched-r2-v1`
- Scope: primary/replay checkpoint identity bookkeeping
- Does not change: data, model topology, initialization, optimization,
  checkpoint tensors, predictions, metrics, thresholds, promotion gates, or
  gameplay claims

## Context

The four ADR 0176 roles completed their frozen 3,000-step protocol. For each
arm, the primary and rotated-host replay produced byte-identical model tensors,
byte-identical optimizer tensors, identical validation prediction probes, and
identical scientific metrics.

The terminal classifier nevertheless marked role-neutral scientific identity
false. Its comparison retained `checkpoint_manifest_blake3`. That manifest
covers `state.json`, and `state.json` records elapsed wall time. Primary and
replay therefore had different manifest hashes even though their trainable and
optimizer state was identical:

- component model tensor:
  `65198601d91f99ebc99783ecb9844ccbcbdcfa11d2e53d7806e76f4aa398cdb8`;
- component optimizer tensor:
  `c254a5c3fec4a8f798a1b59728d3c99bb3f0d91b09d5ddc8fc9c6c0fbaa36c5b`;
- the only component manifest-file difference was the `state.json` checksum,
  whose payload differed in `elapsed_seconds`.

Elapsed wall time is runtime evidence, not scientific identity. Treating its
transitive checksum as a reproducibility gate made deterministic cross-host
replay impossible by construction.

## Decision

Role-neutral primary/replay comparison removes only
`optimization.checkpoint_manifest_blake3`.

The comparison continues to require:

1. identical authorization, cache, protocol, graph, parameter count, layout,
   and initial tensor identity across all roles;
2. identical global step and training-example count within each replay pair;
3. byte-identical final parameter tensors;
4. identical validation prediction probes; and
5. equality of every other role-neutral scientific-identity field.

The full checkpoint manifest remains recorded in each immutable role report.
It is excluded only from the cross-host equality predicate.

## Verification

The repair must pass:

- a regression where primary and replay differ only in checkpoint-manifest
  hash and still satisfy replay integrity;
- the existing regression where a final tensor mismatch defeats integrity;
- the complete ADR 0176 test suite;
- Ruff and Python compilation checks; and
- deterministic reclassification of the four already-frozen reports.

No role is retrained and no metric is recomputed.

## Consequences

ADR 0176 receives an honest integrity result. Its scientific conclusion does
not change: component-anatomy supervision is not promoted because validation
total correlation and pairwise log loss both regress against the scalar
control, despite a small MAE improvement.
