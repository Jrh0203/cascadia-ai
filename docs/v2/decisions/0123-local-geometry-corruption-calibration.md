# ADR 0123: Local Geometry Corruption Calibration

Status: complete

Date: 2026-06-16

Experiment ID: `local-geometry-corruption-calibration-v1`

## Context

ADR 0122 identified local geometry as the dominant train-only specialization
channel. A complete permutation cost the 200-epoch model 32.08 train recall
points but only 1.08 validation points. If ADR 0120 fails, a structural pilot
will regularize this block during training.

This nontraining audit calibrates one corruption rate before that pilot. It
cannot change ADR 0120 or authorize training by itself.

## Frozen Arms

Run three independent corruption rates:

- 10% on john1;
- 25% on john3; and
- 50% on john4.

For every tile query, rank items by their immutable `tile_item_hash`. Select
the first `ceil(rate * width)` items, requiring at least two, and cyclically
rotate only local-geometry columns `[8, 188)` among those selected items.
Everything else remains byte-identical.

Replay both the ADR 0116 20-epoch and ADR 0118 200-epoch checkpoints on train
and validation, with and without corruption. Report recall, exact-query
recovery, train-validation recall gap, gap reduction, and validation damage.

## Gates

Every arm must preserve checkpoint/cache identities, cover every query and
item exactly once, remain finite, and keep all closed domains closed.

An arm is feasible only when:

- it reduces the ADR 0118 train-validation recall gap by at least 25%;
- ADR 0118 validation recall damage is at most 0.02; and
- ADR 0116 validation recall damage is at most 0.02.

Select the smallest feasible corruption rate. If no arm passes, classify
`local_geometry_corruption_not_calibrated`; otherwise classify
`local_geometry_corruption_calibrated`.

The selected rate becomes the frozen local-geometry dropout probability for a
future structural-regularization training pilot if ADR 0120 fails. No sweep or
rate adjustment after observing this audit is allowed.

## Cluster Execution

john1, john3, and john4 run the 10%, 25%, and 50% arms concurrently while
john2 continues the sole ADR 0120 origin. john1 combines the reports.

## Maximum Compute

Three independent nontraining two-checkpoint replays, one combined report,
focused and full tests, and documentation. No training, new data, teacher
rollout, sealed test, gameplay, cloud, Modal, or external compute.

## Result

Every checkpoint, cache, coverage, numerical, and closed-domain gate passed.

| Rate | Extended gap reduction | Source validation damage | Extended validation damage | Feasible |
|---:|---:|---:|---:|---:|
| 0.10 | 11.06% | 0.18% | 0.12% | No |
| 0.25 | 24.9948% | 0.42% | 0.34% | No |
| 0.50 | 48.39% | 0.74% | 0.72% | Yes |

The 25% arm remained below the frozen 25% gap-reduction threshold and was
rejected without rounding. The classification is
`local_geometry_corruption_calibrated`; the selected rate is `0.50`.

If ADR 0120 fails, the next structural training pilot must use 50%
local-geometry dropout and may not retune this probability.

Combined scientific BLAKE3:
`e77c61db83f8453aac20ddaf8b4b184c2c1abb43299df0f59750d7002702430a`.
