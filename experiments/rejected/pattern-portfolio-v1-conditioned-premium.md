# Pattern Portfolio V1: Conditioned Premium

Status: rejected on 2026-06-11.

## Frozen Policy

`pattern-portfolio-v1-k8-h6-b8-m4-t2-conditioned-premium` retained promoted
one-turn opportunity `O1` and added only the exact first-rotation incremental
premium `C2-C1`. With one future personal turn it is exactly pattern-aware;
on the final personal turn opportunity is exactly zero.

## Pilot

Seeds 26300-26309, sequential:

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| Paired mean delta | +0.025 | >= +0.500 | Fail |
| Bear delta | -0.575 | >= +0.500 | Fail |
| Total wildlife delta | -0.075 | >= 0.000 | Fail |
| Non-Bear wildlife delta | +0.500 | >= -0.500 | Pass |
| Habitat delta | +0.025 | >= -0.500 | Pass |
| Runtime per treatment game | 2.866s | <= 5.000s | Pass |

The 95% paired confidence interval was -1.291 to +1.341 and the record was
4-0-6. Nature Tokens improved by 0.075.

## Conclusion

The anchor cleanly removed the full competition policy's non-Bear collapse,
but it also removed the Bear and total-score gains. This closes scalar
interpolation between promoted one-turn opportunity and the exact conditioned
two-turn objective as a promising hand-authored route. No confirmation was
run.

Artifacts:

- `docs/archive/v2/reports/pattern-portfolio-v1-runtime-smoke-1.json`
  (`85cec0e6372eb28913bcab90c4fb07db670a2cc6122b3d45a29868d6b265e440`)
- `docs/archive/v2/reports/pattern-portfolio-v1-pilot10.json`
  (`d01fb9c65391fe58da183a1b158a9829a6107e05f5d6c70506437ef41dc49e8c`)
