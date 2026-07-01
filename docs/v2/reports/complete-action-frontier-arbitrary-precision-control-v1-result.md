# Complete-Action Frontier Arbitrary-Precision Control V1 Result

Classification: `arbitrary_precision_control_invalid`.

ADR 0105 attempted to reconstruct the frozen first 24 scale-16 box optima with 96-digit Decimal arithmetic and a breakpoint active-set derivation. It did not call the float64 analytic or projected solvers. Sealed test, gameplay, new teacher compute, cloud, and external compute remained closed.

## Numerical Result

- Aggregate target recall: 99.53%.
- Exact target sets: 83.33%.
- Maximum normalization residual: `1.81E-94`.
- Maximum Decimal KKT violation: `1.05E-95`.
- Maximum objective difference from frozen float64 analytic: `0.01126553805012420853074322153`.
- Maximum offset difference from frozen float64 analytic: `0.05742618510120569688234968345`.

| Group | Origin | Replay | Recall | Exact | Objective difference | KKT |
|---:|---|---|---:|---:|---:|---:|
| 0 | john1 | john3 | 97.30% | no | `0.004547104571156175156181406977` | `5E-98` |
| 1 | john2 | john1 | 100.00% | yes | `0.002731299268194093651530653566` | `2.5E-97` |
| 2 | john3 | john2 | 100.00% | yes | `0.005723929616092667647693073259` | `1.5E-97` |
| 3 | john4 | john1 | 100.00% | yes | `0.008203538069904381972805881505` | `1E-97` |
| 4 | john1 | john4 | 100.00% | yes | `0.004114538536742256258069566824` | `3.5E-97` |
| 5 | john2 | john1 | 100.00% | yes | `0.001191407191960553609757835274` | `2.5E-97` |
| 6 | john3 | john2 | 100.00% | yes | `0.0001174239731429450815194830859` | `2E-96` |
| 7 | john4 | john3 | 100.00% | yes | `0.002388295870890890917808115576` | `2.5E-97` |
| 8 | john1 | john3 | 97.67% | no | `0.0006126012674325641571522335318` | `5E-98` |
| 9 | john3 | john4 | 100.00% | yes | `0.008266052600541067721807682335` | `2.5E-97` |
| 10 | john2 | john1 | 100.00% | yes | `0.0009277578047810987407751693083` | `1.5E-96` |
| 11 | john4 | john2 | 96.88% | no | `0.01126553805012420853074322153` | `1.5E-96` |
| 12 | john1 | john3 | 100.00% | yes | `0.001439456599196459790279199760` | `2E-96` |
| 13 | john3 | john1 | 100.00% | yes | `0.0004720087374247394615839077912` | `1.5E-96` |
| 14 | john1 | john3 | 97.30% | no | `0.001680082622976931753502240750` | `6.5E-97` |
| 15 | john3 | john1 | 100.00% | yes | `1.845784412026447157229343335E-14` | `8.5E-96` |
| 16 | john1 | john3 | 100.00% | yes | `0.001261508090387150443810484519` | `1.05E-96` |
| 17 | john3 | john2 | 100.00% | yes | `0.0002285260768252590603073028622` | `1E-97` |
| 18 | john2 | john4 | 100.00% | yes | `0.0004765890088455085795515522962` | `1.5E-96` |
| 19 | john4 | john2 | 100.00% | yes | `0.004742000905215529853756465040` | `1E-97` |
| 20 | john2 | john4 | 100.00% | yes | `0.005874120294731762121262946144` | `5.5E-97` |
| 21 | john4 | john2 | 100.00% | yes | `0.002443322042294085831650526998` | `1.05E-95` |
| 22 | john2 | john4 | 100.00% | yes | `0.008916543167788723888137588471` | `2E-96` |
| 23 | john4 | john2 | 100.00% | yes | `0.005117383071439018125690225973` | `5E-98` |

## Frozen Gates

| Gate | Result |
|---|---|
| `all_24_exact_target_sets` | fail |
| `all_24_group_gates_passed` | fail |
| `all_24_replays_identical` | pass |
| `all_origin_and_replay_resources_passed` | pass |
| `control_pipeline_passed` | fail |

## Failure Cause

The preregistration specified exact integer rank conversion, but the frozen expected-rank targets are fractional float64 values. The implementation therefore truncated target ranks before computing probabilities. Normalization and KKT residuals are excellent for that altered objective, but 23 of 24 groups differ from the frozen float64 objective. This is an input-conversion pipeline error, not evidence against the active-set derivation.

## Dynamic Cluster Throughput

- Origin critical path: 3.24 seconds.
- End-to-end origin plus confirmation: 5.96 seconds.
- Scheduled group-process time: 79.83 seconds.
- Confirmation compute fraction: 59.59%.
- Mean active group processes: 13.39; peak: 23.
- Group-process occupancy relative to 40 physical cores: 33.47%.
- Idle process-slot seconds while compatible work was queued: 0.00.
- Duplicate discovery fraction: 0.00%; every origin group was unique and every duplicate was an explicit cross-host replay.
- Source identity: 111 files, `4ba1989ca0f7eb0ab1989fa3aa0ff189bc236380280a0cfdd95fa137eca221c8`, identical on john1, john2, john3, john4.

| Host | Tasks | Scheduled seconds | Final capacity |
|---|---:|---:|---:|
| john1 | 12 | 16.64 | 8 |
| john2 | 13 | 17.14 | 8 |
| john3 | 12 | 25.67 | 8 |
| john4 | 11 | 20.38 | 6 |

## Authorized Successor

Preregister one corrected replay using `Decimal.from_float` for every frozen expected-rank value. Reuse the active-set method, dynamic scheduler, and frozen evidence unchanged. No optimizer or model treatment is authorized.
