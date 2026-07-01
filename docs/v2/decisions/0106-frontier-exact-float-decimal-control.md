# ADR 0106: Frontier Exact-Float Decimal Control

Status: completed as `frozen_optimizer_hyperparameters_insufficient`; sealed
test and gameplay closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-exact-float-decimal-control-v1`

## Context

ADR 0105's independent active-set derivation reached normalization residual
`1.81e-94` and KKT violation `1.05e-95`, but its preregistered integer rank
conversion changed the frozen objective. The expected-rank cache stores
fractional float64 values; 23 of 24 integerized objectives differed from the
ADR 0103 analytic evidence.

ADR 0105 is permanently invalid. This ADR changes only the conversion of each
expected-rank input from integer truncation to exact binary-float preservation
with `Decimal.from_float`.

## Frozen Inputs And Method

- ADR 0103 combined report BLAKE3:
  `dd5f1fee29a1ef93ab96da97303143b5f0aedf82afe340838e4eb06b096522c0`.
- ADR 0103 analytic scientific BLAKE3:
  `6ecfbee0e5dbac42f8853aefc142a8e641b26554b98f5b7c470e9dd7dd446e75`.
- ADR 0105 combined report BLAKE3:
  `40c8d2c7f0480912ca8a2211a3a9c0be1095e3fe4cd933ce140a54c370b2bbd0`.
- Cohort digest:
  `30899dec701f053d96023f963b473681516fb0df00a58edf54146c623fd2769d`.
- The same first 24 groups, scale-16 target, student temperature 2,
  `screen +/- 12` box, champion-frontier anchors, width 64, and stable
  action-hash tie break.
- The same 96-digit `ROUND_HALF_EVEN` Decimal arithmetic, breakpoint
  active-set derivation, Decimal normalization, KKT, objective, selector, and
  frozen analytic comparisons.

The sole numerical correction is:

`rank_i = Decimal.from_float(float(frozen_rank_i))`.

No integer conversion, decimal string round trip, float64 analytic solver,
projected solver, objective, gradient, KKT helper, or frontier selector may
enter the corrected scientific path.

## Gates

All ADR 0105 gates remain unchanged:

- all 24 groups have normalization residual at most `1e-60`;
- all 24 groups have Decimal KKT violation at most `1e-60`;
- objective and normalization-offset differences from ADR 0103 are each at
  most `1e-12`;
- active lower/interior/upper counts match exactly;
- group identity, candidate count, target slots, target hits, exact-set result,
  and winner retention match exactly;
- aggregate target recall and exact sets are both 100%;
- all Decimal values are finite;
- every origin and replay stays below 4 GiB RSS with zero process swaps and no
  attributable positive system-swap growth;
- all four source bundles match;
- every group receives one bit-identical replay on a different host; and
- sealed test, gameplay, teacher, cloud, and external compute remain closed.

## Dynamic Cluster Protocol

Reuse ADR 0105's manifest-backed one-group scheduler:

- 24 unique origin tasks;
- origin priority across john1-john4;
- resource-qualified ramp from two toward at most 10 processes per host;
- replay dependency released immediately when its origin completes;
- replay host must differ from origin host;
- no coarse host barrier;
- atomic outputs and missing-task-only resume.

Record queue snapshots, per-host capacity, active processes, queued-compatible
idle slot seconds, task counts, critical path, host-seconds, confirmation
fraction, duplicate discovery fraction, memory, and swap.

## Mechanical Classification

1. `exact_float_decimal_control_invalid`
   - any numerical, identity, resource, replay, or sealed-domain gate fails.
2. `frozen_optimizer_hyperparameters_insufficient`
   - every gate passes. ADR 0103's frozen analytic optimum passes while frozen
     free-AdamW remains at 59.22% recall and zero exact sets.

A passing result authorizes exactly one calibrated local optimizer mechanism.
It does not authorize a representation change or full trainer directly.

## Maximum Compute

Exactly 24 corrected one-group origins, 24 corrected cross-host replays, one
source identity per host, focused/full tests, and one combined report. No
extra group, rank conversion, precision, solver, threshold, seed, optimizer,
model, trainer, validation treatment, sealed test, gameplay, cloud, Modal, or
external compute.

## Result

All 24 corrected one-group origins and all 24 cross-host replays completed.
Every scientific payload reproduced bit-identically, source identity matched
across john1-john4, peak process RSS remained below 886 MiB, process swaps
remained zero, and no task recorded attributable positive system-swap growth.

The exact-float Decimal control passed every numerical and selector gate:

- 100% recall of all 851 target slots;
- 100% exact recovery across all 24 target sets;
- maximum normalization residual `1.65e-94`;
- maximum Decimal KKT violation `9e-96`;
- maximum objective difference from the frozen float64 analytic solution
  `1.887e-14`; and
- maximum normalization-offset difference `4.686e-13`.

The dynamic queue completed the 48 origin/replay tasks in 5.93 seconds,
peaked at 22 concurrent group processes, used 72.67 scheduled process-seconds,
recorded zero idle process-slot seconds while compatible work was queued, and
assigned 9-14 tasks per host. Duplicate discovery remained zero.

Under the frozen precedence, the classification is
`frozen_optimizer_hyperparameters_insufficient`. ADR 0103's exact objective,
residual box, and independent Decimal control all recover every target set,
while frozen free-parameter AdamW reaches only 59.22% recall after 1,200
updates with zero exact sets.

Exactly one calibrated local optimizer mechanism is now authorized before any
representation change or full trainer.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-exact-float-decimal-control-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-exact-float-decimal-control-v1-result.md`.
