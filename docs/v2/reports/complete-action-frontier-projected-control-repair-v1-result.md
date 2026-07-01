# Complete-Action Frontier Projected-Control Repair V1 Result

Classification: `projected_control_repair_invalid`.

ADR 0104 changed only the independent projected control's maximum iteration count from 10,000 to 100,000 on the frozen first 24 ADR 0103 groups. The frozen analytic, free-AdamW, and neural evidence was not rerun. Sealed test, gameplay, new teacher compute, cloud, and external compute remained closed.

## Numerical Result

| Shard | Host | Groups converged | Recall | Exact sets | Max KKT | Max objective gap |
|---:|---|---:|---:|---:|---:|---:|
| 0 | john1 | 3/6 | 99.10% | 83.33% | 3.927e-09 | 1.632e-09 |
| 1 | john2 | 4/6 | 100.00% | 100.00% | 9.194e-09 | 8.727e-09 |
| 2 | john3 | 3/6 | 87.98% | 66.67% | 3.851e-09 | 8.566e-08 |
| 3 | john4 | 2/6 | 100.00% | 100.00% | 4.931e-09 | 2.067e-10 |

- Aggregate recall: 96.83%.
- Aggregate exact-set recovery: 87.50%.
- Maximum projected KKT violation: `9.194e-09` against `1e-8`.
- Maximum absolute objective gap: `8.566e-08` against `1e-7`.

## Frozen Gates

| Gate | Result |
|---|---|
| `all_24_groups_converged` | fail |
| `all_24_selections_match_analytic` | fail |
| `kkt_gate_passed` | pass |
| `objective_gap_gate_passed` | pass |
| `repair_pipeline_passed` | fail |

Both decision-tolerance numerical gates passed. The campaign is still invalid because 12 of 24 groups did not reach the stricter `1e-9` stopping tolerance and three groups selected a different target set despite tiny objective gaps. The preregistered optimizer treatment is therefore not authorized.

## Cross-Host Replays

| Shard | Origin | Replay | Scientific BLAKE3 |
|---:|---|---|---|
| 0 | john1 | john2 | `e7f386a76898e06336fbf270103f6b7f04d0c250757c4486233c2b2b4b17a61c` |
| 1 | john2 | john3 | `c8666b3894c8f2abc40ffafcb0b6e877259557c3826610057aea0e20feaec981` |
| 2 | john3 | john4 | `96ea99147540215a78613fac2623c97c9b82cc75302a6c75592382717fa56cbd` |
| 3 | john4 | john1 | `8c79df34ffab3646743036c298edcf2ac794a483f95ed9dce28beb615600a022` |

Every origin/replay scientific payload was bit-identical.

## Campaign Throughput

- Origin critical path: 152.63 seconds.
- End-to-end origin plus confirmation: 297.55 seconds.
- Scheduled shard time: 929.19 host-seconds.
- Confirmation compute fraction: 50.27%.
- Origin physical-core occupancy from worker CPU intervals: 15.85%; allocated six-worker occupancy was 26.42%.
- End-to-end physical-core occupancy from worker intervals: 16.49%.
- Duplicate discovery fraction: 0.00%; origins solved disjoint groups and replay work was explicit confirmation.
- Peak parent RSS: 887.8 MiB; peak worker RSS: 50.5 MiB.
- Process swaps zero: true; attributable system swap growth absent: true.
- Source identity: 110 files, `44f13ddbeac1648b83dc1f9efaca7c4dcbd7efb103863c9fa478d1da510eb9ad`, identical on john1, john2, john3, john4.

The worker traces show severe runtime skew: several groups finished in under five seconds while the longest took more than 150 seconds. Static six-group host shards therefore left cores idle near each barrier. Future independent-group campaigns should use smaller resumable work units and a shared dynamic queue across john1-john4.

## Authorized Successor

Preregister one independent arbitrary-precision reconstruction of the frozen analytic optimum and selector. It must use a separate high-precision derivation, retain the same 24 groups, and replay across hosts. More projected iterations, threshold relaxation, and model treatments remain unauthorized.
