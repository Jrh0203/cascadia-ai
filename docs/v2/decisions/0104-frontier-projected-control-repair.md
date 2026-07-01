# ADR 0104: Frontier Projected-Control Precision Repair

Status: completed as `projected_control_repair_invalid`; sealed test and
gameplay closed.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-projected-control-repair-v1`

## Context

ADR 0103 was mechanically invalid because its independent accelerated
projected control stopped at 10,000 iterations with maximum KKT violation
`3.304e-8` against `1e-8` and maximum objective gap `2.622e-7` against
`1e-7`. The control already reached 96.47% recall and 79.17% exact sets, and
its origin/replay payloads were bit-identical.

Every other result remains frozen:

- exact box-constrained optimum: 100% recall and 100% exact sets;
- free-parameter AdamW at 1,200 updates: 59.22% recall and 0% exact sets;
- full neural continuation at 1,200 exposures: 58.45% recall and 0% exact
  sets; and
- all seven ADR 0103 origin/replay payloads: bit-identical.

This ADR repairs only numerical convergence of the independent control. It is
not a new objective, optimizer treatment, model treatment, or discovery
campaign.

## Frozen Inputs

- ADR 0103 combined report BLAKE3:
  `dd5f1fee29a1ef93ab96da97303143b5f0aedf82afe340838e4eb06b096522c0`.
- Analytic scientific BLAKE3:
  `6ecfbee0e5dbac42f8853aefc142a8e641b26554b98f5b7c470e9dd7dd446e75`.
- Free-AdamW scientific BLAKE3:
  `b77ba8a385438d7173e209a7c4c9e60a9de6d87968ecaded9145af702c3cef3a`.
- The exact first 24 ADR 0103 cohort groups, selected checkpoint, scale-16
  cache, residual box, student temperature, selected-score initialization,
  and accelerated projected-gradient implementation.
- Source bundle will be frozen identically on john1-john4 before launch.

No ADR 0103 analytic, free-AdamW, or neural work may be rerun or changed.

## Corrected Control

Increase only the projected-control maximum iteration count from 10,000 to
100,000. Keep:

- initial step 8;
- exact float64 objective and gradient;
- box projection at `screen +/- 12`;
- monotone restart and backtracking;
- KKT stop tolerance `1e-9`;
- analytic objective and selection comparison; and
- the same first 24 groups.

Split the 24 groups into four ordered six-group shards:

| Host | Origin shard | Group indices |
|---|---:|---|
| john1 | 0 | 0-5 |
| john2 | 1 | 6-11 |
| john3 | 2 | 12-17 |
| john4 | 3 | 18-23 |

Each host may use up to six worker processes because groups are independent.
Workers must report per-process RSS, swaps, convergence, iteration count,
KKT violation, objective gap, and selection metrics. Output order remains the
frozen cohort order.

## Replay Ring

After the origin wave, replay each complete shard on the next host:

- shard 0: john1 to john2;
- shard 1: john2 to john3;
- shard 2: john3 to john4;
- shard 3: john4 to john1.

Every scientific payload must be bit-identical after excluding elapsed time.

## Gates

The repaired control passes only when:

- all 24 groups converge before or at 100,000 iterations;
- maximum projected KKT violation is at most `1e-8`;
- maximum absolute objective gap from the frozen analytic solution is at most
  `1e-7`;
- projected target hits and exact-set outcomes match the analytic report on
  all 24 groups;
- every score and objective is finite;
- every worker remains below 4 GiB RSS with zero swaps;
- no host records attributable positive system-swap growth;
- all four source identities and all four replay payloads match; and
- sealed test, gameplay, teacher, cloud, and external compute remain closed.

## Mechanical Classification

1. `projected_control_repair_invalid`
   - any gate above fails.
2. `frozen_optimizer_hyperparameters_insufficient`
   - every repaired control gate passes, because ADR 0103's frozen analytic
     optimum passes and frozen free-AdamW result fails.

A passing repair authorizes exactly one calibrated local optimizer mechanism.
It does not authorize a representation change or full trainer directly.

## Cluster And Throughput Contract

Launch all four origin shards concurrently, then all four ring replays
concurrently. This is disjoint CPU work in the origin wave and justified
confirmation in the replay wave. Record physical-core occupancy from the
cluster history, critical path, host-seconds, worker scaling, replay fraction,
and duplicate discovery fraction.

## Maximum Compute

Exactly four six-group origins, four six-group replays, focused/full tests,
source checks, and one combined report. No extra group, iteration treatment,
seed, optimizer, model, trainer, validation treatment, sealed test, or
gameplay run.

## Result

All four origin shards and all four ring replays completed. Source identity
matched across john1-john4, every origin/replay scientific payload was
bit-identical, peak worker RSS remained below 51 MiB, and no process or host
recorded attributable positive swap growth.

The increased iteration budget repaired the two decision-tolerance numerical
gates. Maximum projected KKT violation was `9.194e-9` against `1e-8`, and
maximum objective gap was `8.566e-8` against `1e-7`. Aggregate target recall
reached 96.83% and exact-set recovery reached 87.50%.

The full preregistered repair still failed. Only 12 of 24 groups reached the
stricter `1e-9` stopping tolerance by 100,000 iterations, and groups 4, 15,
and 16 selected different target sets from the analytic solution despite tiny
objective gaps. Under the frozen precedence, the classification is
`projected_control_repair_invalid`; no optimizer or model treatment is
authorized.

The four disjoint origins completed on a 152.63-second critical path. The
complete origin-plus-confirmation campaign took 297.55 seconds and 929.19
scheduled host-seconds. Confirmation consumed 50.27% of scheduled compute and
duplicate discovery remained zero. Static six-group host shards exposed severe
runtime skew, so future independent-group campaigns must use finer resumable
units and dynamic cross-host claiming rather than waiting at coarse host
barriers.

The only authorized successor is a separately preregistered,
arbitrary-precision reconstruction of the frozen analytic optimum and
selector. It must use an independent derivation on the same 24 groups with
cross-host replay. More projected iterations, threshold relaxation, and model
treatments remain closed.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-projected-control-repair-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-projected-control-repair-v1-result.md`.
