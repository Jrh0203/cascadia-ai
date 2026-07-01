# ADR 0120: Conditional Tile Optimizer Schedule

Status: preregistered

Date: 2026-06-16

Experiment ID: `conditional-tile-optimizer-schedule-v1`

## Context

ADR 0118 held the conditional tile architecture, features, balanced top-32
membership objective, AdamW optimizer, `3e-4` learning rate, weight decay,
batch size, initialization, seed, caches, and checkpoint selection fixed while
increasing exposure from 20 to 200 epochs. The model nearly memorized the full
train cache at 99.80% target recall, but validation recall fell from 70.59% to
67.75%. Uniform exposure is insufficient.

ADR 0119 independently found that uniform query sampling does not pass the
frozen target-mass mismatch gate. Its mechanical branch therefore authorizes
exactly one optimizer-schedule treatment before this ranker route is closed.

The treatment tests one hypothesis: fixed `3e-4` updates after the original
20-epoch horizon produce harmful late movement, while preserving the useful
early trajectory and annealing only late updates may improve generalization.

## Frozen Evidence

- ADR 0118 classification:
  `extended_exposure_tile_insufficient`.
- ADR 0118 combined scientific BLAKE3:
  `3e01e3b0cc1d55f54f3ec880deb0459a3ef09609d6594714ce3cd78578f7e555`.
- ADR 0118 selected weights BLAKE3:
  `7acd245b20bf5a35bb3bcab848f3b4b3014d763058fa803b0b4ae3b17c80205d`.
- ADR 0119 classification for sampling:
  `uniform_query_sampling_not_explanatory`.
- ADR 0119 combined scientific BLAKE3:
  `c569aff575ab5e300335f35bca2108d9d7a22a679ede8d6b8a80a888cbfeb7eb`.
- Train cache payload BLAKE3:
  `1707fd84fac77dee0e4878165bf8f8b98869b6d4d206deb55db030321cc96ede`.
- Validation cache payload BLAKE3:
  `b128a3b5bf53e135febf39dba02d9c7486692245523516a5ee3031eea795229b`.

Sealed test, gameplay, new teacher compute, cloud, Modal, and external compute
remain closed.

## Frozen Treatment

Train exactly one conditional tile ranker from scratch with the complete
ADR 0118 contract, except for this epoch-level learning-rate schedule:

- epochs 1 through 20: constant `3e-4`;
- epochs 21 through 200: one cosine decay from `3e-4` to `3e-6`; and
- no restart, warmup, cycle, plateau detector, validation feedback, or other
  schedule component.

The cosine schedule is:

`3e-6 + 0.5 * (3e-4 - 3e-6) * (1 + cos(pi * (epoch - 20) / 180))`

for epochs 21 through 200.

Everything else remains frozen: `HierarchicalFactorRanker`, observable parent
state, query context, item features, hidden width 256, retrieval width 32,
balanced target-membership BCE, AdamW, weight decay `1e-4`, batch size 32,
seed `2026061648`, immutable train and validation caches, 200 epochs, and
train-only checkpoint selection by target recall then exact-query recovery.

No warm start, early stop, target-mass resampling, architecture change, feature
change, loss change, optimizer-family change, width change, regularization
change, second seed, or schedule sweep is allowed. Validation is evaluated
once after the 200th epoch.

## Evaluation

The selected checkpoint must:

1. contain a complete 200-epoch trajectory with the frozen learning rate on
   every event;
2. replay bit-identically on a different host;
3. remain finite, below 4 GiB peak process RSS, and at zero process swaps;
4. replace only the tile checkpoint in the frozen ADR 0115 hierarchy;
5. hold draft, wildlife, and the final selector oracle-perfect for the mixed
   ceiling; and
6. preserve all cache, source, closed-domain, and coverage identities.

## Gates

Classify `optimizer_schedule_tile_sufficient` only if:

- train tile factor recall exceeds 95%;
- validation tile factor recall exceeds 90%;
- oracle-other-stage validation target-action recall exceeds 98%;
- oracle-other-stage validation R4800 winner retention exceeds 98%; and
- the integrated learned proposal passes every ADR 0115 proposal gate.

Classify `optimizer_schedule_tile_insufficient` when the pipeline passes but
any strength gate fails. Classify `optimizer_schedule_pipeline_invalid` before
interpreting strength when any identity, schedule, numerical, coverage,
replay, resource, or sealed-domain gate fails.

A sufficient result freezes this tile proposal and opens the separately
preregistered complete-action selector. An insufficient result closes this
conditional pointwise tile ranker under uniform exposure, target-mass
resampling, and optimizer scheduling. No further epoch, learning-rate,
schedule, or sampling sweep is authorized.

## Cluster Execution

- john2 owns the sole 200-epoch MLX origin.
- john3 owns source identity, schedule-trajectory validation, and cross-host
  replay.
- john4 owns the oracle-other-stage mixed ceiling.
- john1 owns implementation, tests, reporting, and integrated evaluation.

The origin is the only duplicate-sensitive training job. During it, every
other host must claim a distinct open-data diagnostic or implementation task
from the research queue; preparation alone is not a reason to remain idle.
Dependent replay, ceiling, and integration launch immediately after verified
checkpoint fanout.

## Maximum Compute

One 200-epoch full-cache origin, one cross-host replay, one mixed-stage
ceiling, one integrated hierarchy evaluation, independent nonduplicative
open-data diagnostics on the remaining hosts, focused and full tests, one
report, and documentation. No second origin, seed sweep, schedule sweep,
teacher rollout, sealed test, gameplay, cloud, Modal, or external compute.
