# Complete-Action Frontier Exact-Float Decimal Control V1 Result

Classification: `frozen_optimizer_hyperparameters_insufficient`.

ADR 0106 preserved every frozen fractional expected-rank bit with `Decimal.from_float`, then solved the first 24 scale-16 box optima with the independent 96-digit breakpoint active-set derivation. The scientific path did not call the float64 analytic or projected solvers. Sealed test, gameplay, new teacher compute, cloud, and external compute remained closed.

## Numerical Result

- Aggregate target recall: 100.00%.
- Exact target sets: 100.00%.
- Maximum normalization residual: `1.65E-94`.
- Maximum Decimal KKT violation: `9E-96`.
- Maximum objective difference from frozen float64 analytic: `1.887161393571969686576708694E-14`.
- Maximum offset difference from frozen float64 analytic: `4.685680289509035768594375052E-13`.

| Group | Origin | Replay | Recall | Exact | Objective difference | KKT |
|---:|---|---|---:|---:|---:|---:|
| 0 | john1 | john3 | 100.00% | yes | `7.448382330501573189074073999E-15` | `1E-96` |
| 1 | john2 | john1 | 100.00% | yes | `3.937286002752357168600524572E-15` | `2E-97` |
| 2 | john3 | john4 | 100.00% | yes | `3.368771960210624037132342117E-15` | `5E-98` |
| 3 | john4 | john1 | 100.00% | yes | `3.895445853747886105800240545E-15` | `5E-97` |
| 4 | john1 | john2 | 100.00% | yes | `7.975663872401489369678651277E-15` | `8E-97` |
| 5 | john2 | john4 | 100.00% | yes | `2.717427248162086085696339593E-15` | `6E-97` |
| 6 | john3 | john1 | 100.00% | yes | `9.051782326608048646770829696E-15` | `1E-96` |
| 7 | john4 | john2 | 100.00% | yes | `4.661397612560937556144221622E-15` | `6E-97` |
| 8 | john1 | john4 | 100.00% | yes | `1.406749306971249160909703603E-15` | `1.5E-97` |
| 9 | john3 | john1 | 100.00% | yes | `8.356252467287725446943198288E-15` | `1E-97` |
| 10 | john4 | john3 | 100.00% | yes | `3.945514691211659996804042546E-15` | `1.5E-96` |
| 11 | john2 | john1 | 100.00% | yes | `3.887611370821149957940510573E-15` | `5E-97` |
| 12 | john1 | john3 | 100.00% | yes | `1.080594767765728496909768056E-14` | `2E-96` |
| 13 | john3 | john2 | 100.00% | yes | `8.592241410642489434758355860E-15` | `5E-97` |
| 14 | john4 | john1 | 100.00% | yes | `4.762196913875871475055318497E-15` | `4E-97` |
| 15 | john1 | john3 | 100.00% | yes | `1.887161393571969686576708694E-14` | `7.6E-96` |
| 16 | john3 | john4 | 100.00% | yes | `1.648496700610923534668257112E-14` | `1.5E-97` |
| 17 | john4 | john3 | 100.00% | yes | `5.137856109720505639671568539E-15` | `1.0E-96` |
| 18 | john1 | john4 | 100.00% | yes | `7.904157131724621478654655148E-16` | `5E-97` |
| 19 | john3 | john4 | 100.00% | yes | `7.209020456647641527360777657E-15` | `4.5E-97` |
| 20 | john4 | john3 | 100.00% | yes | `9.311365740317133488888728498E-15` | `1.0E-96` |
| 21 | john2 | john1 | 100.00% | yes | `5.939230647064149275332366973E-15` | `9E-96` |
| 22 | john2 | john1 | 100.00% | yes | `3.991432071978016094358750752E-15` | `2.5E-96` |
| 23 | john2 | john3 | 100.00% | yes | `3.680814112705506245242558234E-15` | `5E-98` |

## Frozen Gates

| Gate | Result |
|---|---|
| `all_24_exact_target_sets` | pass |
| `all_24_group_gates_passed` | pass |
| `all_24_replays_identical` | pass |
| `all_origin_and_replay_resources_passed` | pass |
| `control_pipeline_passed` | pass |

## Dynamic Cluster Throughput

- Origin critical path: 3.05 seconds.
- End-to-end origin plus confirmation: 5.93 seconds.
- Scheduled group-process time: 72.67 seconds.
- Confirmation compute fraction: 57.58%.
- Mean active group processes: 12.25; peak: 22.
- Group-process occupancy relative to 40 physical cores: 30.64%.
- Idle process-slot seconds while compatible work was queued: 0.00.
- Duplicate discovery fraction: 0.00%; every origin group was unique and every duplicate was an explicit cross-host replay.
- Source identity: 112 files, `0ca0470812733a5b1ff670d587335acf081a3200e69cc67f499913efc924dac3`, identical on john1, john2, john3, john4.

| Host | Tasks | Scheduled seconds | Final capacity |
|---|---:|---:|---:|
| john1 | 14 | 18.40 | 8 |
| john2 | 9 | 12.23 | 6 |
| john3 | 13 | 21.13 | 8 |
| john4 | 12 | 20.90 | 8 |

## Authorized Successor

The independent exact-float numerical control passes. ADR 0103 therefore authorizes exactly one calibrated local optimizer mechanism before any representation change or full trainer.
