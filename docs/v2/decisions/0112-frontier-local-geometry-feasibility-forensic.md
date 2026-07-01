# ADR 0112: Frontier Local-Geometry Feasibility Forensic

Status: completed as `parameterized_fit_or_optimizer_insufficient`; one
same-representation mechanistic control is authorized.

Date: 2026-06-16

Experiment ID:
`complete-action-frontier-local-geometry-feasibility-forensic-v1`

## Context

ADR 0111 passed every pipeline gate but its frozen-base local-geometry adapter
reached only 71.13% target recall and 25% exact target sets. Group 2 fit
exactly while groups 0, 1, and 3 numerically converged below the strength gate.
The single representation treatment authorized by ADR 0110 is exhausted.

Before considering any mechanistically distinct successor, the retained group
inputs can determine whether the failure is forced by the adapter's bounded
output range or exact observable aliasing, or instead arises from the learned
parameter sharing and optimization path.

## Frozen Evidence

- ADR 0111 combined report BLAKE3:
  `6e21675a7c05ac815368f1cc02c2b6769bfddbd57571d21efaa695fa2752e15f`.
- ADR 0111 source bundle BLAKE3:
  `68aacab18627326a7880fc55c43c8de8bcaccc25c5cee7cb148679e9ab03c6d1`.
- The same four groups, selected model, exact 13-relation local features,
  canonical action, prior/global inputs, target mask, selector, residual
  bounds, and correction range.

No training result may be reconstructed. No gradient, optimizer update,
parameter fit, target, representation, selector, range, group, or metric may
change.

## Audit

For each unrotated group:

1. Run one frozen selected-model inference and reconstruct the exact ADR 0111
   local adapter input rows.
2. Hash every valid row's exact float32 bytes and partition candidates into
   exact observable equivalence classes.
3. Count duplicate classes, mixed target/non-target classes, and exact
   score-tie contradictions.
4. Measure selected-base target recovery and residual saturation.
5. Compute each candidate's exact reachable score interval under
   `clip(base_residual + correction, -12, 12)` for correction in
   `[-12, 12]`.
6. Compute the candidate-independent interval ceiling by assigning every
   target its upper endpoint and every non-target its lower endpoint, then
   running the frozen deployed selector. This assignment dominates every
   other independent bounded assignment for target recovery.

If no equivalence class mixes target and non-target candidates, the
candidate-independent ceiling is also the exact observable-class ceiling.
If mixed classes exist, aliasing remains material and unresolved by this
audit; do not invent independent corrections inside those classes.

No ADR 0111 adapter weights, accepted-rate histories, or terminal score rows
may be fabricated from summary statistics.

## Execution

Run groups 0-3 as four distinct origin tasks across john1-john4 and replay
each on a different host. Use the dynamic one-process-per-host queue and
record source identity, payload identity, resource use, wall time,
process-seconds, occupancy, and queued-work idle.

## Gates

The audit pipeline passes only when:

- all four frozen groups are present exactly once;
- selected model, dataset, cache, cohort, and ADR 0111 evidence identities
  match;
- every feature, residual, interval, and score is finite;
- interval lower endpoints never exceed upper endpoints;
- selector accounting is exact;
- all four cross-host replay payloads are bit-identical;
- source identity matches on john1-john4;
- peak RSS is below 4 GiB with zero process swaps and no attributable positive
  system-swap growth; and
- training, gradients, optimizer updates, validation, sealed test, gameplay,
  new teacher compute, cloud, and external compute remain closed.

## Mechanical Classification

1. `local_geometry_feasibility_forensic_invalid`
   - any identity, finite, interval, replay, resource, or sealed gate fails.
2. `bounded_adapter_output_insufficient`
   - the valid candidate-independent interval ceiling misses 90% aggregate
     target recall or 75% exact target sets.
3. `exact_observable_aliasing_material`
   - the independent ceiling passes but at least one exact feature class mixes
     target and non-target candidates.
4. `parameterized_fit_or_optimizer_insufficient`
   - the independent ceiling passes and no exact feature class mixes target
     and non-target candidates.

No outcome directly authorizes a second representation treatment or full
trainer. The valid classification may authorize one separately preregistered
mechanistic control that does not change the frozen representation.

## Maximum Compute

Exactly four frozen-inference origins, four cross-host replays, source checks,
focused/full tests, one combine, and one report. No training, gradients,
optimizer updates, adapter reconstruction, representation treatment, full
trainer, validation treatment, sealed test, gameplay, cloud, Modal, or
external compute.

## Result

All four frozen-inference origins and cross-host replays completed with
bit-identical scientific payloads. Source identity matched across john1-john4,
all resource gates passed, and the audit used no training, gradients, or
optimizer updates.

Across 11,087 candidates:

- every exact local adapter input row was unique;
- no target/non-target observable class collision existed;
- no selected-base residual was saturated at either bound; and
- assigning each target its reachable upper score and each non-target its
  reachable lower score recovered 100% recall and 100% exact target sets in
  every group.

The selected base itself recovered 30.28% aggregate target recall. The exact
bounded interval ceiling recovered 100%. The frozen representation and output
range therefore contain enough information and score authority; ADR 0111's
remaining failure lies in the shared parameterized fit or its optimization
path.

The mechanical classification is
`parameterized_fit_or_optimizer_insufficient`. This authorizes one separately
preregistered mechanistic control using the same frozen representation. It
does not authorize a second representation treatment or full trainer.

The campaign completed in 3.07 seconds, scheduled 10.17 process-seconds,
averaged 3.32 active processes, peaked at four, and recorded zero queued-work
idle.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-local-geometry-feasibility-forensic-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-local-geometry-feasibility-forensic-v1-result.md`.
