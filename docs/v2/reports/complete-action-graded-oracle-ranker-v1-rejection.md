# Complete-Action Graded Oracle Ranker V1 Rejection

Status: **rejected on validation; sealed test and gameplay closed unopened**

## Verdict

The corrected observable-only MLX experiment completed all three frozen replicas,
the preregistered cross-host validation matrix, and selected-model performance
checks on john1, john2, and john3. The john2 replica won the frozen selection
objective, but it failed every overall, phase, and subset winner-recall gate.
ADR 0082 was therefore not authorized and no test group or gameplay seed was opened.

## Replica Selection

| Train host | Seed | Cross host | Retained regret | Top-64 recall | R4800 MAE |
|---|---:|---|---:|---:|---:|
| john1 | 2026061601 | john2 | 0.098733 | 71.67% | 1.422790 |
| john2 | 2026061602 | john3 | 0.090184 | 73.33% | 1.447013 |
| john3 | 2026061603 | john1 | 0.105826 | 72.92% | 1.435484 |

Selected checkpoint: `step-000003592-epoch-0008-batch-000000` from john2.

## Validation

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Overall top-64 R4800 winner recall | >98% | 73.33% | Fail |
| Mean retained R4800 regret | <0.15 | 0.090184 | Pass |
| Early recall | >=97% | 69.05% | Fail |
| Middle recall | >=97% | 65.48% | Fail |
| Late recall | >=97% | 87.50% | Fail |
| Nature-token subset recall | >=95% | 74.35% | Fail |
| Independent-draft subset recall | >=95% | 71.43% | Fail |

Every regret, finite-score, complete-group, latency, memory, and swap gate passed.

## Selected-Model Performance

| Host | Action scores/s | P99 decision ms | Peak RSS MiB | Swap delta |
|---|---:|---:|---:|---:|
| john1 | 102,405 | 79.68 | 346.4 | -8388608 |
| john2 | 102,124 | 79.79 | 530.3 | 0 |
| john3 | 101,888 | 79.88 | 364.1 | 0 |

## Diagnosis

- The learned screen reduced retained regret from 0.113024 to 0.090184, a 20.2% reduction.
- Exact-winner recall moved only from 71.67% to 73.33%.
- Cross-host metrics were bit-identical and selected-model inference passed on
  every Mac, so this is a target/learning result rather than an execution or
  portability failure.
- The frozen gates are not weakened post hoc. The next experiment must revise
  the oracle or target design; K2048 and a large self-play launch remain closed.

## Protocol Closure

- Test authorization file: absent.
- Sealed-test evaluation output: absent.
- Test groups read by this reporter: no.
- ADR 0082: closed unopened.
- ADR 0083: closed unopened.

Machine-readable evidence is in
`docs/v2/reports/complete-action-graded-oracle-ranker-v1-rejection.json`.
