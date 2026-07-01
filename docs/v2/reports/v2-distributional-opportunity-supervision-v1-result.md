# V2 Distributional Opportunity Supervision v1 Result

**Completed:** 2026-06-17
**Experiment:** `v2-distributional-opportunity-supervision-v1`
**Classification:** `distributional_opportunity_factorial_null`

## Verdict

The matched distributional factorial is a replicated offline null.

G1 and Q2 learned real uncertainty structure, improved CRPS and pairwise
probability calibration, and reduced top-action regret. Neither produced an
informative enough winner confidence set under the frozen gates. E3 failed
CRPS, probability, uncertainty, and coverage gates.

No arm was selected. Test, final, gameplay, and successor training remained
closed.

## Validation

The open validation split contained 512 groups, 2,048 candidates, and 24,576
shared-seed continuations.

| Arm | CRPS delta vs C0 | Regret delta vs C0 | Top-value recall delta | Winner-set coverage | Mean set size | Eligible |
|---|---:|---:|---:|---:|---:|---|
| G1 Gaussian | -0.0531 | -0.0352 | +0.0059 | 0.9941 | 3.6563 | no |
| **Q2 quantile** | **-0.0454** | **-0.1012** | **+0.0234** | 0.9902 | **3.5977** | **no** |
| E3 CRPS atoms | +0.4204 | -0.0075 | +0.0020 | 0.3750 | 1.0000 | no |

Q2 was directionally strongest for choice quality, reducing mean top-action
regret from `0.6755` to `0.5742`. Its confidence set still contained 3.60 of
four candidates on average and therefore failed the preregistered
informativeness gate. G1's set was even wider at 3.66.

The result supports using uncertainty as a diagnostic for teacher error and
search allocation. It does not support replacing the point-value head or
opening gameplay.

## Exactness

All four primary arms and all four rotated-host replays matched exactly on:

- final model file;
- final parameter tensor;
- prediction probe;
- scientific identity.

The compact exact sparse-R2 representation was used throughout; no 441-cell
state was materialized.

## Artifacts

- terminal classification:
  `artifacts/experiments/v2-distributional-opportunity-supervision-v1/classification.json`;
- all primary and replay runs:
  `artifacts/experiments/v2-distributional-opportunity-supervision-v1/runs`;
- frozen authorization:
  `artifacts/experiments/v2-distributional-opportunity-supervision-v1/control/authorization-package/authorization.json`.
