# Complete-Action Local-Geometry Ranker V1 Rejection

Status: **rejected on validation; sealed test and gameplay closed unopened**

## Verdict

All three preregistered MLX replicas completed, each selected checkpoint was
replayed on another Mac, and all origin/cross scientific payloads were
bit-identical. The john2 replica won the frozen selection order and passed
every integrity, regret, throughput, latency, memory, and swap gate. It
nevertheless missed every overall winner-recovery gate, every phase recall
and coverage gate, and both subset recall gates. ADR 0088 is rejected without
opening sealed test data or gameplay.

## Replica Selection

| Train host | Seed | Cross host | Epochs | Retained regret | Top-64 recall | R4800 MAE |
|---|---:|---|---:|---:|---:|---:|
| john1 | 2026061601 | john3 | 7 | 0.113024 | 71.67% | 4.912167 |
| john2 | 2026061602 | john3 | 15 | 0.093757 | 74.17% | 1.721006 |
| john3 | 2026061603 | john1 | 7 | 0.113024 | 71.67% | 4.912167 |

Selected checkpoint: `step-000004045-epoch-0009-batch-000000` from john2.

## Frozen Gates

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Overall exact winner recall | >98% | 74.17% | Fail |
| Overall confidence-set coverage | >=99% | 87.92% | Fail |
| Distinguishable-winner recall | >=98% | 88.16% | Fail |
| Mean retained R4800 regret | <0.15 | 0.093757 | Pass |
| Early exact recall | >=97% | 72.62% | Fail |
| Early confidence coverage | >=98% | 92.86% | Fail |
| Middle exact recall | >=97% | 69.05% | Fail |
| Middle confidence coverage | >=98% | 82.14% | Fail |
| Late exact recall | >=97% | 81.94% | Fail |
| Late confidence coverage | >=98% | 88.89% | Fail |
| Nature-token exact recall | >=95% | 75.39% | Fail |
| Independent-draft exact recall | >=95% | 76.19% | Fail |

Every phase and subset retained-regret gate passed.

## Performance

| Replay | Action scores/s | P99 decision ms | Peak RSS MiB | Swap delta |
|---|---:|---:|---:|---:|
| john1-origin | 51,796 | 172.57 | 390.2 | -698152386 |
| john2-origin | 81,740 | 97.86 | 573.7 | 0 |
| john3-origin | 80,565 | 98.46 | 497.6 | 0 |
| john1-cross-on-john3 | 82,249 | 98.44 | 564.8 | 0 |
| john2-cross-on-john3 | 80,992 | 98.09 | 397.4 | 0 |
| john3-cross-on-john1 | 43,836 | 181.11 | 390.4 | -530778685 |

## Diagnosis

- Retained regret improved from 0.113024 to 0.093757, a 17.0% reduction.
- Exact-winner recall improved only from 71.67% to 74.17%.
- The treatment is portable and fast; the failure is scientific, not an
  execution artifact.
- Local geometry alone is not the missing mechanism. The observed R4800
  exceptions motivate hard retention of public champion/frontier anchors
  while learning only the nonfrontier fill.

## Execution

- Initial three-host training launch skew: 0.247 seconds.
- Maximum host-lock queue: 0.000073 seconds.
- john2 resumed once after its SSH wrapper disconnected; checkpoint and
  scientific contracts were unchanged.
- All six origin/cross performance replays passed.

## Protocol Closure

- Test authorization file: absent.
- Sealed-test or gameplay output: absent.
- Test groups read by this reporter: no.
- New teacher compute: not used.
- K2048: not opened.

Machine-readable evidence is in
`docs/v2/reports/complete-action-local-geometry-ranker-v1-rejection.json`.
