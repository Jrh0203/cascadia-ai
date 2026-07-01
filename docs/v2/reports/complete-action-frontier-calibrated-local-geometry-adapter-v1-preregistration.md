# Complete-Action Frontier Calibrated Local-Geometry Adapter V1 Preregistration

Date: 2026-06-16

Experiment ID: `complete-action-frontier-calibrated-local-geometry-adapter-v1`

## Question

Can the strongest established public-observable local relation construction
close the four-group frontier fit gap when isolated as a zero-initialized
adapter over the frozen selected model and trained with the calibrated
monotone optimizer?

## Frozen Treatment

- Same four groups, targets, scale-16 expected-rank objective, rotation cycle,
  selected base model, 1,200-update ceiling, and calibrated AdamW constants.
- Frozen base predictions; only one ADR 0088-shaped local adapter is trainable.
- Inputs: exact 13-relation rotation-canonical local geometry, canonical
  action, prior features, and global features.
- Correction: `clip(base + 12*tanh(adapter), -12, 12)`.
- Zero-initialized head must reproduce the frozen base exactly.
- Corrected convergence tests improvement only in the eligible rate domain
  while retaining all below-floor diagnostics.

## Cluster Execution

Four distinct group origins run concurrently across john1-john4. Every group
is then replayed on a different host. The dynamic queue schedules ready work
without fixed host barriers, runs at most one MLX optimizer process per host,
resumes missing outputs only, and records wall time, process-seconds, mean and
peak active processes, and idle slot-seconds while compatible work is queued.

## Decision Rule

- Invalid pipeline:
  `calibrated_local_geometry_pipeline_invalid`.
- Valid pipeline below 90% terminal recall or 75% exact sets:
  `calibrated_local_geometry_insufficient`.
- Valid pipeline meeting both terminal gates:
  `calibrated_local_geometry_sufficient`.

Only the sufficient outcome authorizes one bounded full-trainer pilot. A
failure consumes ADR 0110's single representation treatment and does not
authorize another treatment.

## Closed Domains

No second representation, alternate architecture, sweep, optimizer change,
objective change, full trainer, validation treatment, sealed test, gameplay,
new teacher compute, cloud, Modal, or external compute.
