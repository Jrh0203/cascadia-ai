# Complete-Action Frontier Boundary Ranking V1 Rejection

Status: **rejected on open validation**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-boundary-ranking-v1`

## Verdict

The conservative smooth weakest-target versus strongest-nontarget objective
is rejected for the shared neural scorer. It has a perfect bounded score-space
ceiling, but decreasing neural training loss moves the deployed top-K set
backward.

## Launch Evidence

| Audit | Result |
|---|---:|
| Widest real group | 10,854 actions |
| Target gradient signs | 32 of 32 correct |
| Nontarget gradient signs | 10,790 of 10,790 correct |
| Excluded gradients | all zero |
| Full-model update | finite, 424 MB RSS, zero swaps |
| Widest validation groups | 12 |
| Direct initial target recall | 36.41% |
| Direct final target recall | 100% |
| Direct initial exact sets | 0% |
| Direct final exact sets | 100% |
| Maximum residual used | 9.403 of 12 |

The objective and score range are sufficient when scores are independent.

## Selected Result

No trained epoch beat the initial checkpoint.

| Metric | Selected | Gate |
|---|---:|---:|
| Train target recall | 29.36% | 60% |
| Train exact sets | 0.18% | 5% |
| Validation target recall | 26.21% | 50% |
| Validation exact sets | 0% | 1% |
| Validation winner recall | 76.67% | 75% |
| Validation confidence coverage | 90.42% | 90% |
| Validation regret | 0.061734 | <0.15 |

Four substantive gates failed. All integrity, finite-score, memory, swap, and
sealed-domain gates passed.

## Trajectory

Boundary loss fell every epoch, from 3.2696 to 3.0191. Validation target recall
never recovered its 26.21% starting point and ended at 18.76%. Exact target-set
recovery remained zero throughout; final winner recall was 57.92% and final
confidence coverage was 81.25%.

This closes single-extreme smooth boundary optimization. The next objective
must distribute pressure across the whole width-32 nonfrontier fill boundary,
using rank-matched target/hard-negative pairs so every required slot receives
direct learning signal.

## Execution

- john2 training wall: 780.70 seconds including host-lock lifecycle.
- john1 open evaluation: 70.47 seconds.
- john3 maximum-width audit: 1.30 seconds.
- john4 bounded score-space audit: 3.82 seconds.
- Runtime source: 91 files, byte-identical across all four Macs.
- Full Python gate: 232 tests; Ruff clean.
- Process swaps: zero.
- Sealed test, gameplay, second seed, new teacher compute, and external
  compute: unopened.

Machine-readable evaluation:
`artifacts/experiments/complete-action-frontier-boundary-ranking-v1/runs/john2-seed-2026061606/open-evaluation.json`.
