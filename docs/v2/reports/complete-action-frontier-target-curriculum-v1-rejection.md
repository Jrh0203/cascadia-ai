# Complete-Action Frontier Target Curriculum V1 Rejection

Status: **rejected on open validation**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-target-curriculum-v1`

## Verdict

Target-only fine-tuning did not fit the set target. Uniform set cross entropy
is rejected as the optimization surrogate for exact width-64 retention.

## Selected Result

| Metric | Warm start | Selected pilot | Gate |
|---|---:|---:|---:|
| Train target recall | 29.36% | 30.97% | 60% |
| Train exact sets | 0.18% | 0.18% | 5% |
| Validation target recall | 26.21% | 26.29% | 50% |
| Validation exact sets | 0% | 0% | 1% |
| Validation winner recall | 76.67% | 74.58% | 75% |
| Validation confidence coverage | 90.42% | 90.00% | 90% |
| Validation regret | 0.061734 | 0.065729 | <0.15 |

The selected checkpoint was epoch 1. Six later epochs failed to improve
target recall; epoch 7 reached only 24.41% even though target-only training
loss continued to decline.

## Mechanism

The result closes three simpler explanations:

- Exact observable collision mass is zero.
- The production ±12 residual range is ample: ±6 can recover every target
  set, and ±3 recovers 99.88% of validation target slots.
- Removing the opposed R1200 listwise and screen-only terms did not improve
  target allocation.

The remaining optimization mismatch is the surrogate. Uniform cross entropy
rewards average target probability across roughly 32 positives and thousands
of negatives, but exact deployment succeeds only when the weakest retained
target clears the strongest excluded nontarget. The next objective must act on
that boundary directly.

## Execution

- john2 training wall: 907.14 seconds including host-lock lifecycle.
- john1 open evaluation: 70.32 seconds.
- john3 reachability audit: 7.91 seconds.
- john4 trajectory audit: 0.09 seconds.
- Runtime source bundle: 88 files, byte-identical on all four Macs.
- Process swaps: zero.
- Sealed test, gameplay, second seed, new teacher compute, and external
  compute: unopened.
