# ADR 0108: Frontier Monotone AdamW Stop-Rule Repair

Status: completed as `free_stage_passed`; ADR 0107 neural Stage 2 authorized.
Full trainer, sealed test, and gameplay remain closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-monotone-adamw-stop-repair-v1`

## Context

ADR 0107's calibrated monotone AdamW passed its free-residual strength gate at
96.59% recall and 79.17% exact sets. Its pipeline was invalid because groups
0, 2, 8, 14, and 23 reached float32 numerical saturation before 1,200
accepted updates. Every remaining proposal was finite but failed the
same-batch monotonicity test after 16 halvings.

Requiring meaningless updates below float32 resolution is a stop-rule defect,
not an optimizer treatment. This ADR repairs only that rule.

## Frozen Evidence

- ADR 0107 free combined report BLAKE3:
  `f550f552a14400ffe6f33ec1b3cacea355c0b138651f815a5767336455b1b184`.
- ADR 0107 source bundle BLAKE3:
  `1ded82bb44d6d43cd0e5ac097d68c6621f763bcf7409e44114f985b363206546`.
- Frozen completed groups:
  `1,3,4,5,6,7,9,10,11,12,13,15,16,17,18,19,20,21,22`.
- Repair groups: `0,2,8,14,23`.
- The exact ADR 0107 model, objective, optimizer direction, maximum rate,
  moments, weight decay, backtracking factor, trial count, groups, checkpoint
  schedule, and source inputs.

No completed group, optimizer constant, objective, target, model,
representation, dataset, or strength gate may change.

## Corrected Stop Rule

For a repair group, perform the unchanged update loop. When no proposal is
accepted:

- all 16 proposals must have finite loss and finite parameters;
- current parameters, first moment, second moment, direction, and loss must be
  finite;
- the smallest attempted rate must be below `1e-7`;
- no proposal may improve current loss by more than the frozen `1e-12`
  tolerance; and
- at least one prior update must have been accepted.

If every condition holds, record `numerically_converged` and retain the last
accepted parameters and metrics. Otherwise the group fails. Numerical
convergence counts as valid completion; it does not fabricate additional
updates.

## Execution

Run exactly the five repair groups as independent origins through the
one-MLX-process-per-host dynamic queue, then replay each on a different host.
Merge their terminal reports with the frozen 19 ADR 0107 groups in original
cohort order.

The recombined Stage 1 passes only when:

- aggregate recall is at least 95%;
- aggregate exact sets are at least 75%;
- all 24 groups either complete 1,200 accepted updates or satisfy the repaired
  numerical-convergence rule;
- every score, loss, parameter, moment, direction, and rate is finite;
- all origin/replay scientific payloads are bit-identical;
- source identity matches on john1-john4;
- peak RSS stays below 4 GiB with zero process swaps and no attributable
  positive system-swap growth; and
- sealed test, gameplay, teacher, cloud, and external compute remain closed.

## Mechanical Classification

1. `monotone_adamw_stop_repair_invalid`
   - any repair, merge, source, replay, resource, finite, or sealed gate fails.
2. `calibrated_optimizer_mechanism_insufficient`
   - the repaired pipeline is valid but misses the free strength gate.
3. `free_stage_passed`
   - every pipeline and strength gate passes.

Only `free_stage_passed` authorizes ADR 0107 neural Stage 2 with the same
optimizer mechanism. It does not authorize a full trainer directly.

## Maximum Compute

Exactly five origins, five cross-host replays, source checks, focused/full
tests, one merge, and one report. No rerun of the frozen 19 groups, alternate
stop threshold, optimizer treatment, learning rate, group, budget, model,
representation, neural Stage 2 before the gate, validation treatment, sealed
test, gameplay, cloud, Modal, or external compute.

## Result

All five origins and five cross-host replays completed. Every replay scientific
payload was bit-identical. All five groups met the corrected numerical
convergence rule:

| Group | Accepted updates | Recall | Exact | Smallest attempted rate |
|---:|---:|---:|---:|---:|
| 0 | 916 | 100.00% | yes | `5.899e-12` |
| 2 | 1,105 | 100.00% | yes | `2.949e-12` |
| 8 | 767 | 97.67% | no | `2.949e-12` |
| 14 | 871 | 100.00% | yes | `5.899e-12` |
| 23 | 834 | 100.00% | yes | `7.373e-13` |

Each convergence event evaluated exactly 16 finite proposals, retained finite
parameters, moments, direction, and loss, and observed zero candidate
improvement at float32 resolution.

The recombined 24-group result retained 96.59% recall and 79.17% exact sets.
Every lineage, finite, resource, source, replay, sealed-boundary, pipeline, and
strength gate passed. The mechanical classification is `free_stage_passed`.

The sparse dynamic queue completed ten tasks in 6.71 seconds, scheduled 22.55
MLX process-seconds, averaged 3.36 active processes, peaked at four, and
recorded zero idle process-slot seconds while compatible work was queued.
Source identity matched across john1-john4 at 114 files and SHA-256
`ab432f3768d89642b93e25019fd078db5dafa16fefe1636a84fb30f29dbd4903`.

ADR 0107 neural Stage 2 is now authorized with the unchanged optimizer:
exactly four origins and four cross-host replays. No full trainer, validation
treatment, sealed test, or gameplay is authorized.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-monotone-adamw-stop-repair-v1/reports/free-combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-monotone-adamw-stop-repair-v1-result.md`.
