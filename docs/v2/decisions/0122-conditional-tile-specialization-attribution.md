# ADR 0122: Conditional Tile Specialization Attribution

Status: complete

Date: 2026-06-16

Experiment ID: `conditional-tile-specialization-attribution-v1`

## Context

ADR 0121 proved late-fit margin specialization without material exact aliases
or broad covariate shift. From the 20-epoch to 200-epoch checkpoint, normalized
train boundary margin improved by 1.7033 while validation worsened by 1.1260.
If ADR 0120 fails, the successor must use structural regularization.

The tile item vector has three frozen semantic blocks:

- tile-factor identity: columns `[0, 8)`;
- local geometry: columns `[8, 188)`; and
- descendant summaries: columns `[188, 249)`.

This audit identifies which block acquired train-only predictive association.
It performs no training and cannot alter ADR 0120.

## Frozen Treatment

For each semantic block, independently replay the ADR 0116 20-epoch and
ADR 0118 200-epoch checkpoints on train and validation under:

1. the exact unmodified cache; and
2. one deterministic cyclic within-query permutation of only that block.

For query index `q` with width `w > 1`, rotate the selected block by
`1 + q mod (w - 1)`. Other columns, query context, group state, item order,
labels, widths, and selectors remain unchanged.

Report target recall and exact-query recovery before and after permutation.
For each block define:

`specialization contribution = (extended train drop - source train drop) -
(extended validation drop - source validation drop)`.

## Gates

Every arm must preserve cache and checkpoint identities, score every query and
item exactly once, remain finite, and keep all closed domains closed.

Classify `specialization_block_identified` when:

- the largest specialization contribution is at least 0.05; and
- it exceeds the second-largest contribution by at least 0.02.

Otherwise classify `specialization_distributed_across_blocks`.

The identified block selects targeted feature-block dropout or corruption in a
future separately preregistered structural-regularization pilot. A distributed
result selects global capacity or weight regularization. No result authorizes
another exposure, sampling, or learning-rate schedule treatment.

## Cluster Execution

- john1 owns tile-factor identity.
- john3 owns local geometry.
- john4 owns descendant summaries.
- john1 combines the three independent arms.

All three arms launch concurrently while john2 continues the sole ADR 0120
origin. There is no duplicate arm or training.

## Maximum Compute

Three independent two-checkpoint open-cache replays, one combined report,
focused and full tests, and documentation. No training, new data, teacher
rollout, sealed test, gameplay, cloud, Modal, or external compute.

## Result

Every cache, checkpoint, coverage, numerical, and closed-domain gate passed.

| Feature block | Specialization contribution |
|---|---:|
| Tile-factor identity | +0.0457 |
| Local geometry | +0.2446 |
| Descendant summaries | +0.1056 |

Local geometry exceeded the frozen 0.05 floor and beat the runner-up by
0.1390, well above the 0.02 separation gate. The classification is
`specialization_block_identified` and the selected block is
`local_geometry`.

The mechanism is sharply train-specific. Local-geometry permutation reduced
ADR 0118 train recall by 32.08 points but validation recall by only 1.08
points. The corresponding ADR 0116 drops were 7.67 and 1.14 points. If
ADR 0120 fails, the next structural pilot must target local-geometry feature
dropout or corruption while leaving every other input block and evaluation
contract fixed.

Combined scientific BLAKE3:
`eda9da8157b23535bdee1a266ba0f5fa2e67fc00c5941f8254773310bb9e1f08`.
