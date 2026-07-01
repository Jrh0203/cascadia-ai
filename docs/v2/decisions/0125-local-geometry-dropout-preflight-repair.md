# ADR 0125: Local-Geometry Dropout Preflight Repair

Status: complete

Date: 2026-06-16

Experiment ID:
`conditional-tile-local-geometry-dropout-preflight-repair-v1`

Parent experiment:
`conditional-tile-local-geometry-dropout-preflight-v1`

## Context

The first ADR 0124 preflight closed as
`local_geometry_dropout_preflight_invalid`. Contract, 200-epoch coverage,
cross-host selection digest, and gradient arms all passed. The resource arm
failed two frozen gates:

- peak process RSS was 4,716,576,768 bytes, above 4 GiB; and
- preparation-only overhead was 367.66%, above 50%.

The failure is an implementation defect. The resource harness decompressed and
retained all seven cache shards simultaneously. The training batch path also
copied each query's item matrix once for corruption and again into the packed
batch. Neither behavior is required by the frozen corruption semantics.

The failed preflight combined scientific BLAKE3 is
`1b996efaf6351886d72043c4b76326798dce8b6ac40a2df750685c16522688b2`.

## Frozen Repair

Change only implementation and measurement mechanics:

1. Stream one immutable cache shard at a time in the resource benchmark.
2. Copy source items directly into the already-required packed batch and
   rotate selected local blocks there, eliminating the intermediate query
   copy.
3. Replace full sorting of every query's 64-bit keys with exact partition
   selection. Determine the partition cutoff, include every key below it,
   resolve cutoff ties by original item position, then order the selected
   subset by key and position.

The third change must be mathematically identical to the original lexicographic
full sort. It may not change selected items, selected ordering, corruption
values, dropout count, salt derivation, rate, seed, feature columns, epoch
variation, or any scientific gate.

No model architecture, loss, optimizer, schedule, training seed, data,
checkpoint selection, validation, inference, or branch condition changes.

## Verification

Repeat all four open-train-cache preflight arms using the repaired immutable
source:

- john1: contract, then resource;
- john3: complete 200-epoch coverage;
- john4: gradient-channel audit;
- john1: remote collection and combination.

In addition to the original gates:

- the epoch-one selection digest must equal the original passing digest
  `87a234b381161f78eeefc63199dac85ba342492ed79cee060204a8f36516ed4e`;
- optimized partition selection must equal the frozen full-sort reference over
  deterministic unit cases; and
- all original contract and gradient values must remain semantically valid.

Classify `local_geometry_dropout_preflight_passed` only when every original
gate and the digest-preservation gate pass. Otherwise classify
`local_geometry_dropout_preflight_invalid` and repair only the remaining
implementation defect.

## Maximum Compute

Four repeated open-train-cache preflight arms, one collection, one
combination, focused and full tests, documentation, and immutable source
snapshots. No training, validation, test, gameplay, teacher rollout, cloud,
Modal, or external compute.

## Result

Every repaired arm and combined gate passed.

- Contract and coverage independently reproduced the original selection
  digest
  `87a234b381161f78eeefc63199dac85ba342492ed79cee060204a8f36516ed4e`.
- Complete 200-epoch coverage selected every item between 33.5% and 67.5% of
  exposures, with exact mean 50.0904%.
- The gradient arm reproduced finite, nonzero local and nonlocal input
  gradients and a nonzero parameter-gradient change.
- Streamed median baseline preparation was 0.9287 seconds; treatment
  preparation was 1.1252 seconds, or 21.16% overhead.
- Peak process RSS was 2,033,909,760 bytes with zero swaps.

Classification: `local_geometry_dropout_preflight_passed`.

Combined scientific BLAKE3:
`2b6eacd04b490e3305e10c4603bf42363fdb78f1a8d21cd7f766eeb2441c99e3`.

The ADR 0124 branch is implementation-ready but remains closed until ADR 0120
finishes valid and insufficient.
