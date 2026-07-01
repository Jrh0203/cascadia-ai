# ADR 0105: Frontier Arbitrary-Precision Independent Control

Status: completed as `arbitrary_precision_control_invalid`; sealed test and
gameplay closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-arbitrary-precision-control-v1`

## Context

ADR 0104 increased the frozen projected control from 10,000 to 100,000
iterations. Its maximum KKT violation (`9.194e-9`) and objective gap
(`8.566e-8`) passed their decision tolerances, but only 12 of 24 groups reached
the stricter `1e-9` stop rule and groups 4, 15, and 16 selected different
target sets from the analytic solution. ADR 0104 therefore closed as
`projected_control_repair_invalid`.

Increasing iterations again would not provide independent evidence. This ADR
reconstructs the same convex optimum with arbitrary-precision arithmetic and a
separate active-set derivation. It changes no objective, target, box, model,
optimizer, training data, or deployment selector.

## Frozen Inputs

- ADR 0103 combined report BLAKE3:
  `dd5f1fee29a1ef93ab96da97303143b5f0aedf82afe340838e4eb06b096522c0`.
- ADR 0103 analytic scientific BLAKE3:
  `6ecfbee0e5dbac42f8853aefc142a8e641b26554b98f5b7c470e9dd7dd446e75`.
- ADR 0104 combined report BLAKE3:
  `dbf73a580f996413c80f0196ecf728e2aca613654b4ae1c06fb9c43ce7319bba`.
- Cohort digest:
  `30899dec701f053d96023f963b473681516fb0df00a58edf54146c623fd2769d`.
- The exact first 24 ADR 0103 groups, scale-16 cache, residual box
  `screen +/- 12`, student temperature 2, target scale 16, champion-frontier
  anchors, width 64, and action-hash tie break.
- The frozen ADR 0103 analytic group summaries are comparison evidence only.

No ADR 0103 analytic, free-AdamW, neural, or ADR 0104 projected work may be
rerun or changed.

## Independent Decimal Reconstruction

Use only Python's standard-library `decimal.Decimal` with:

- precision 96 decimal digits;
- `ROUND_HALF_EVEN`;
- `Decimal.from_float` for every frozen float64 screen input;
- exact integer rank conversion; and
- no call to the float64 analytic solver, projected solver, objective,
  gradient, KKT helper, or frontier selector.

For each group:

1. Independently compute positive target weights
   `exp(-(rank - 1) / 16)` and normalize them at 96-digit precision.
2. Set every zero-mass eligible action to its lower score bound.
3. For positive mass `p_i`, use the KKT form
   `score_i = clip(2 * ln(p_i) + c, lower_i, upper_i)`.
4. Solve `c` with an independent breakpoint sweep. At each lower or upper
   active-set transition, maintain the interior target mass and summed bound
   exponentials. Inside a fixed interval solve
   `c = 2 * ln(bound_sum / (1 - interior_mass))` directly.
5. Recompute normalization, KKT residual, and cross entropy in Decimal.
6. Reimplement the width-64 anchored selector by sorting on descending
   Decimal score and ascending action-hash bytes.

Each group records the Decimal objective, offset, normalization residual, KKT
residual, active-set counts, target recovery, winner retention, selected hash
digest, elapsed time, RSS, and swap telemetry. Decimal outputs are serialized
as canonical strings.

## Gates

The independent control passes only when all 24 groups satisfy:

- normalization residual at most `1e-60`;
- Decimal KKT violation at most `1e-60`;
- objective difference from the frozen ADR 0103 analytic summary at most
  `1e-12`;
- normalization-offset difference at most `1e-12`;
- identical active lower, interior, and upper counts;
- identical group identity, candidate count, target slots, target hits,
  exact-set result, and winner-retention result;
- 100% target-positive recall and 100% exact target sets;
- finite Decimal values;
- peak RSS below 4 GiB and zero process swaps;
- no attributable positive system-swap growth;
- identical source bundles on john1-john4;
- one bit-identical scientific replay of every group on a different host; and
- sealed test, gameplay, teacher, cloud, and external compute remain closed.

## Dynamic Cluster Protocol

The unit of work is one group, not a fixed six-group host shard.

- Place all 24 origin groups in one manifest-backed queue.
- Schedule ready origins to the least-loaded compatible host across
  john1-john4, with deterministic group-index tie breaking.
- Permit up to 10 group processes per host after a no-swap smoke.
- Give origin work priority. As origins finish, enqueue their replay on any
  different host and use replay work to backfill free slots.
- Persist atomic per-group outputs and scheduler events. Resume only missing
  tasks after interruption.
- Do not wait for a host-level origin barrier before launching compatible
  confirmation work.

Record useful worker occupancy, queue depth, critical path, host-seconds,
per-host task counts, confirmation fraction, duplicate discovery fraction,
idle time while compatible work was queued, and any scheduler correction.

## Mechanical Classification

1. `arbitrary_precision_control_invalid`
   - any numerical, identity, resource, replay, or sealed-domain gate fails.
2. `frozen_optimizer_hyperparameters_insufficient`
   - every gate passes. ADR 0103's frozen analytic optimum passes while its
     frozen free-AdamW result remains at 59.22% recall and zero exact sets.

A passing result authorizes exactly one calibrated local optimizer mechanism.
It does not authorize a representation change or full trainer directly.

## Maximum Compute

Exactly 24 one-group origins, 24 one-group cross-host replays, one source
identity per host, focused/full tests, and one combined report. No extra
group, precision treatment, solver treatment, projected iteration increase,
threshold change, seed, optimizer, model, trainer, validation treatment,
sealed test, gameplay, cloud, Modal, or external compute.

## Result

All 24 one-group origins and all 24 cross-host replays completed. Every
scientific replay was bit-identical, source identity matched across
john1-john4, peak process RSS remained below 886 MiB, process swaps remained
zero, and no task recorded attributable positive system-swap growth.

The independent Decimal active-set solver was numerically precise for the
objective it received: maximum normalization residual was `1.81e-94` and
maximum KKT violation was `1.05e-95`. That objective was not the frozen
objective. The preregistration specified integer rank conversion, but the
frozen expected-rank inputs are fractional float64 values. The implementation
therefore truncated target ranks before computing probabilities. Twenty-three
of 24 groups differed from the frozen analytic objective, with maximum
objective difference `0.011266` and maximum offset difference `0.057426`.

The altered objective reached 99.53% target recall and 83.33% exact sets, but
those values cannot validate the frozen control. Under the preregistered
precedence, the classification is `arbitrary_precision_control_invalid`; no
optimizer or model treatment is authorized.

The dynamic queue itself passed its operational objective. It completed 48
tasks in 5.97 seconds wall time, reached 23 concurrent group processes, used
79.83 scheduled process-seconds, recorded zero idle process-slot seconds while
compatible work was queued, and assigned 11-13 tasks per host. Duplicate
discovery remained zero; every duplicate task was an explicit cross-host
replay.

The only authorized successor is a new preregistered replay that converts each
frozen expected-rank float with `Decimal.from_float`. It must retain the same
active-set derivation, 96-digit precision, gates, 24 groups, dynamic queue,
and cross-host replay protocol.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-arbitrary-precision-control-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-arbitrary-precision-control-v1-result.md`.
