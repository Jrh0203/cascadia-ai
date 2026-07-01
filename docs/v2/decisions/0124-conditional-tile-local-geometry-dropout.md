# ADR 0124: Conditional Tile Local-Geometry Dropout

Status: contingently preregistered; repaired preflight passed

Date: 2026-06-16

Experiment ID: `conditional-tile-local-geometry-dropout-v1`

Preflight ID: `conditional-tile-local-geometry-dropout-preflight-v1`

## Context

ADR 0120 is the final authorized exposure, sampling, or optimizer-schedule
treatment for the conditional pointwise tile ranker. It remains the active
critical-path experiment and cannot be changed by this ADR.

Independent open-data forensics already froze the only admissible successor if
ADR 0120 is valid but insufficient:

- ADR 0121 found late fit-margin specialization rather than label aliasing or
  material input covariate shift.
- ADR 0122 attributed the largest train-only specialization contribution to
  tile local geometry.
- ADR 0123 selected a 50% within-query local-geometry corruption rate. The
  10% arm removed too little gap, and the 25% arm missed its frozen gate
  without rounding.

The mechanism is targeted structural regularization. During training, prevent
the tile ranker from binding individual candidate identities too tightly to
their local-geometry block while preserving every other observable, the
feature distribution, architecture, objective, optimizer, schedule, and
clean inference contract.

## Branch Authorization

This ADR may launch training only when all of the following are true:

1. ADR 0120 has completed as
   `optimizer_schedule_tile_insufficient`;
2. every ADR 0120 pipeline gate passed;
3. the ADR 0125 repaired preflight completed as
   `local_geometry_dropout_preflight_passed`; and
4. the ADR 0124 manifest was mechanically changed from
   `contingently_authorized` to `authorized` while recording the immutable
   ADR 0120 combined scientific BLAKE3.

If ADR 0120 is sufficient, classify this branch
`not_launched_optimizer_schedule_sufficient` and do not train it. If ADR 0120
is invalid, repair only ADR 0120. Preflight evidence cannot override either
branch.

## Frozen Treatment

Inherit the complete ADR 0120 contract:

- `HierarchicalFactorRanker`;
- tile stage and retrieval width 32;
- hidden width 256;
- balanced top-32 membership BCE;
- AdamW with weight decay `1e-4`;
- batch size 32;
- model and batch-order seed `2026061648`;
- 200 full-cache epochs;
- epochs 1-20 at `3e-4`;
- one cosine decay over epochs 21-200 to `3e-6`;
- train-only checkpoint selection by target recall then exact-query recovery;
- immutable train and validation caches; and
- clean, uncorrupted validation and inference.

Change only the tile item features presented to the training loss:

1. For every cache shard, tile query, and epoch, derive one deterministic
   64-bit salt from dropout seed `2026061650`, the one-based epoch, zero-based
   shard index, and zero-based query index.
2. Read the first eight bytes of each immutable 16-byte `tile_item_hash` as a
   little-endian unsigned integer.
3. Rank items by the frozen SplitMix64-style mixer applied to item prefix XOR
   query salt, with original item position as the deterministic tiebreak.
4. Select exactly `ceil(0.50 * query_width)` items, with a two-item minimum
   when query width is at least two.
5. Cyclically rotate only local-geometry columns `[8,188)` by one position
   among selected items.
6. Leave tile identity columns `[0,8)`, descendant-summary columns
   `[188,249)`, every unselected local block, state, context, labels, masks,
   query order, and optimizer state unchanged.

This mechanism is
`epoch-hash-half-query-local-geometry-rotation-v1`. It is structured
feature-block dropout by within-query reassignment, not elementwise zeroing.
It exactly preserves the selected local-geometry feature multiset and avoids
introducing an out-of-distribution all-zero geometry token.

No corruption is permitted during train-metric checkpoint evaluation,
validation, replay, mixed ceiling, integration, or deployment.

## Preflight

The preflight uses only the open train cache and performs no training:

- **Contract arm on john1:** prove exact selected counts, byte-identical
  nonlocal columns and source arrays, exact selected-block rotation, complete
  query/item coverage, and produce the epoch-one selection digest.
- **Coverage arm on john3:** evaluate all 200 epochs, require exact selected
  counts on every query exposure, no item selected zero or 200 times, every
  item selection rate in `[0.30,0.70]`, exact mean coverage, and independently
  reproduce the epoch-one digest.
- **Gradient arm on john4:** on one fixed 32-query batch, require finite and
  nonzero baseline and treatment parameter gradients, a changed objective and
  parameter gradient, and nonzero treatment input gradients in both local and
  nonlocal feature blocks.
- **Resource arm on john1:** run three complete train-cache batch-preparation
  passes for baseline and treatment, require exact coverage, finite timing,
  median preparation overhead at most 50%, peak process RSS below 4 GiB, and
  zero process swaps.

The combined preflight additionally requires one shared train-cache identity,
cross-host epoch-one digest equality, and all closed domains preserved.
Preflight failure invalidates implementation and authorizes repair only; it
does not authorize a different dropout rate or mechanism.

## Evaluation

If the branch opens, the selected checkpoint must:

1. contain the exact 200-epoch ADR 0120 learning-rate trajectory;
2. record the exact preregistered selected and eligible item counts on every
   epoch;
3. replay bit-identically on a different host under clean inference;
4. remain finite, below 4 GiB peak process RSS, and at zero process swaps;
5. replace only the tile checkpoint in the frozen ADR 0115 hierarchy;
6. hold draft, wildlife, and the final selector oracle-perfect for the mixed
   ceiling; and
7. preserve all cache, source, closed-domain, and coverage identities.

## Gates

Classify `local_geometry_dropout_tile_sufficient` only if:

- train tile factor recall exceeds 95%;
- validation tile factor recall exceeds 90%;
- oracle-other-stage validation target-action recall exceeds 98%;
- oracle-other-stage validation R4800 winner retention exceeds 98%; and
- the integrated learned proposal passes every ADR 0115 proposal gate.

Classify `local_geometry_dropout_tile_insufficient` when the pipeline passes
but any strength gate fails. This closes the conditional pointwise tile
representation and moves the program upstream; no dropout-rate, corruption,
capacity, exposure, sampling, optimizer, or schedule sweep is authorized.

Classify `local_geometry_dropout_pipeline_invalid` before interpreting
strength when any branch, preflight, identity, schedule, corruption coverage,
numerical, replay, resource, integration, or sealed-domain gate fails.

## Cluster Execution

Preflight runs while john2 continues the sole ADR 0120 origin:

- john1: contract and resource arms, then combination;
- john3: all-epoch coverage arm;
- john4: gradient-channel arm.

If the training branch opens:

- john2 owns the sole 200-epoch MLX origin;
- john3 owns source identity and clean cross-host replay;
- john4 owns the oracle-other-stage mixed ceiling; and
- john1 owns integration, reporting, and the next independent portfolio.

The other three hosts must continue nonduplicative decision-changing work
during the origin. No duplicate discovery origin is allowed.

## Maximum Compute

Before the branch decision: four open-train-cache preflight arms, one
combination, focused and full tests, documentation, and source snapshots.

After authorization: one 200-epoch full-cache origin, one clean cross-host
replay, one mixed-stage ceiling, one integrated hierarchy evaluation, one
classification, and independent nonduplicative backfill work. No second
origin, seed sweep, dropout sweep, corruption variant, teacher rollout,
sealed test, gameplay, cloud, Modal, or external compute.

## Preflight Result

The first implementation preflight closed invalid because its resource harness
retained all seven decompressed shards and the training path copied query
features twice. Contract, coverage, digest, and gradient gates passed.

ADR 0125 repaired only those implementation mechanics. The repaired preflight
passed every gate:

- the original epoch-one selection digest was reproduced exactly;
- all 850,246 items were covered across all 9,808 tile queries and 200 epochs;
- per-item selection rates ranged from 33.5% to 67.5%;
- preparation overhead fell from 367.66% to 21.16%;
- peak process RSS fell from 4,716,576,768 to 2,033,909,760 bytes; and
- gradients remained finite, nonzero, and treatment-sensitive.

Repaired preflight scientific BLAKE3:
`2b6eacd04b490e3305e10c4603bf42363fdb78f1a8d21cd7f766eeb2441c99e3`.

ADR 0124 remains contingent on the ADR 0120 result. No ADR 0124 training has
run.
