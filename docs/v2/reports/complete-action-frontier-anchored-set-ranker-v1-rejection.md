# Complete-Action Frontier-Anchored Set Ranker V1

Status: **rejected on validation; sealed test and gameplay closed unopened**

## Verdict

All four preregistered replicas and all origin/cross-host replays completed with bit-identical scientific payloads. The selected replica failed 11 frozen quality gates.

## Replica Selection

| Train host | Seed | Cross host | Epochs | Winner recall | Coverage | Regret |
|---|---:|---|---:|---:|---:|---:|
| john1 | 2026061601 | john3 | 8 | 75.42% | 91.25% | 0.058074 |
| john2 | 2026061602 | john4 | 14 | 76.67% | 90.42% | 0.061734 |
| john3 | 2026061603 | john1 | 11 | 75.00% | 90.83% | 0.061229 |
| john4 | 2026061604 | john2 | 7 | 75.00% | 90.42% | 0.056764 |

Selected checkpoint: `step-000003592-epoch-0008-batch-000000` from john2.

## Frozen Gates

| Metric | Required | Observed |
|---|---:|---:|
| Exact winner recall | >98% | 76.67% |
| Confidence-set coverage | >=99% | 90.42% |
| Distinguishable-winner recall | >=98% | 92.11% |
| Mean retained regret | <0.15 | 0.061734 |
| Target-positive recall | diagnostic | 26.21% |
| Exact target-set recovery | diagnostic | 0.00% |

Failed gates:
- `early_confidence_set_coverage_at_least_0_98`
- `early_top64_recall_at_least_0_97`
- `independent_draft_winner_top64_recall_at_least_0_95`
- `late_confidence_set_coverage_at_least_0_98`
- `late_top64_recall_at_least_0_97`
- `middle_confidence_set_coverage_at_least_0_98`
- `middle_top64_recall_at_least_0_97`
- `nature_token_available_top64_recall_at_least_0_95`
- `top64_confidence_set_coverage_at_least_0_99`
- `top64_distinguishable_winner_recall_at_least_0_98`
- `top64_r4800_winner_recall_strictly_greater_than_0_98`

## Performance

| Replay | Action scores/s | P99 decision ms | Peak RSS MiB | Swap delta |
|---|---:|---:|---:|---:|
| john1-origin | 97,445 | 83.83 | 337.9 | -25165824 |
| john2-origin | 97,578 | 83.87 | 501.9 | 0 |
| john3-origin | 97,566 | 84.19 | 497.0 | 0 |
| john4-origin | 98,289 | 83.71 | 342.4 | 0 |
| john1-cross-on-john3 | 97,733 | 84.08 | 501.5 | 0 |
| john2-cross-on-john4 | 98,806 | 83.22 | 337.1 | 0 |
| john3-cross-on-john1 | 96,547 | 86.74 | 337.5 | 0 |
| john4-cross-on-john2 | 97,736 | 84.03 | 501.2 | 0 |

## Cluster Execution

| Host | Assigned s | Productive s | Queued idle s | Completed | Failed |
|---|---:|---:|---:|---:|---:|
| john1 | 1701.5 | 1339.6 | 0.000 | 5 | 0 |
| john2 | 2031.3 | 1861.1 | 0.000 | 5 | 0 |
| john3 | 1774.9 | 1495.6 | 0.000 | 5 | 0 |
| john4 | 2031.1 | 966.5 | 0.000 | 5 | 0 |

The four training replicas were frozen before the cluster policy moved
to single-host MLX pilots plus independent experiments. John 4's longer
assigned idle interval was a dependency wait for John 2's frozen cross
checkpoint, not unqueued compatible work.

## Diagnosis

- Hard frontier retention improved exact recall by 3.75 percentage points.
- The selected model recovered only 26.21% of target-positive nonfrontier slots and no complete target sets.
- The treatment was portable and fast; the failure is scientific.
- Future discovery should train one MLX pilot while the other Macs run
  independent representation or optimization hypotheses.

## Protocol Boundary

- Origin/cross scientific payloads: bit-identical.
- Test authorization file: absent.
- Sealed-test or gameplay output: absent.
- Test groups read by this reporter: no.

Machine-readable evidence is in the adjacent JSON report.
