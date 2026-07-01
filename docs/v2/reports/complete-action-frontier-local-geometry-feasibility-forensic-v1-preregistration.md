# Complete-Action Frontier Local-Geometry Feasibility Forensic V1 Preregistration

Date: 2026-06-16

Experiment ID:
`complete-action-frontier-local-geometry-feasibility-forensic-v1`

## Question

Did ADR 0111 fail because its bounded local-geometry correction cannot
represent the required target ordering, because exact observable rows require
contradictory corrections, or because the parameterized fit and optimizer did
not realize an otherwise feasible ordering?

## Frozen Audit

- Same four groups, selected model, local input construction, target mask,
  selector, residual range, and correction range.
- One frozen base forward pass per group.
- Exact float32 row hashing for observable equivalence classes.
- Exact candidate score intervals under the frozen clipping equation.
- Target-upper/non-target-lower selector evaluation as the independent bounded
  recovery ceiling.
- No adapter reconstruction, training, gradient, or optimizer update.

## Cluster Execution

Four distinct group origins run across john1-john4 and each is replayed on a
different host. The dynamic queue uses at most one process per host, resumes
missing work only, and records occupancy and queued-work idle.

## Decision Rule

- Invalid audit: `local_geometry_feasibility_forensic_invalid`.
- Independent bounded ceiling below 90% recall or 75% exact sets:
  `bounded_adapter_output_insufficient`.
- Passing independent ceiling with mixed exact feature classes:
  `exact_observable_aliasing_material`.
- Passing independent ceiling without mixed classes:
  `parameterized_fit_or_optimizer_insufficient`.

No outcome directly authorizes a full trainer or second representation
treatment.
