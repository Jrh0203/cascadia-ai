# ADR 0118: Conditional Tile Extended Exposure

Status: complete

Date: 2026-06-16

Experiment ID: `conditional-tile-extended-exposure-v1`

## Context

ADR 0116's unchanged 256-wide target-only tile ranker reached 77.21% train and
70.59% validation recall after 20 full-cache epochs. Its training trajectory
was still improving at the endpoint: train recall rose from 74.18% at epoch 10
to 77.21% at epoch 20.

ADR 0117 then isolated local fit from full-data scale. The same ranker and loss
recovered 100% of targets and exact sets on 16 hard queries, then 100% on 256
hard queries after roughly 200 cohort passes. A two-block self-attention
control was slower and weaker. The mechanical classification was
`full_data_scale_or_optimization_insufficient`.

The next treatment changes only exposure. Architecture, observable features,
conditional query construction, width, objective, optimizer, learning rate,
weight decay, batch size, initialization, seed, caches, and checkpoint
selection remain identical to ADR 0116.

## Frozen Evidence

- ADR 0117 combined scientific BLAKE3:
  `695d4bce6a82047ff73e46a740ae1ef302a6995b9ca6dc14ae82895a22333eae`.
- ADR 0116 combined scientific BLAKE3:
  `0d4681df11a527ca1571008cca8b6e55800866380fb8a4cf7450def1ed54a4f6`.
- ADR 0116 target-only weights BLAKE3:
  `5c13fe87d7b4ac0a8ff9f647f57c69b8d9ab583b3ce2e85e41ee0f3d97e8f514`.
- Train cache payload BLAKE3:
  `1707fd84fac77dee0e4878165bf8f8b98869b6d4d206deb55db030321cc96ede`.
- Validation cache payload BLAKE3:
  `b128a3b5bf53e135febf39dba02d9c7486692245523516a5ee3031eea795229b`.

Sealed test, gameplay, new teacher compute, cloud, and external compute remain
closed.

## Frozen Treatment

Train exactly one conditional tile ranker from scratch with:

- the exact ADR 0116 `HierarchicalFactorRanker`, parent state, query context,
  tile item features, hidden width 256, and retrieval width 32;
- the exact balanced top-32 membership BCE and no auxiliary loss;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- batch size 32 and seed `2026061648`;
- the immutable ADR 0115 train and validation factor caches; and
- 200 epochs instead of 20.

Select the checkpoint only by train target recall, then exact-query recovery.
Evaluate validation once after all 200 epochs. No warm start, early stop,
schedule, curriculum, hard-query resampling, architecture change, feature
change, loss change, width change, or second seed is allowed.

## Evaluation

The selected checkpoint must:

1. replay bit-identically on a different host;
2. remain finite, below 4 GiB peak process RSS, and at zero process swaps;
3. be evaluated with draft, wildlife, and final selector held oracle-perfect;
4. replace only the tile checkpoint in the frozen ADR 0115 hierarchy; and
5. report its full 200-epoch train trajectory and exposure cost.

## Gates

Classify `extended_exposure_tile_sufficient` only if:

- train tile factor recall exceeds 95%;
- validation tile factor recall exceeds 90%;
- oracle-other-stage validation target-action recall exceeds 98%;
- oracle-other-stage validation R4800 winner retention exceeds 98%; and
- the integrated learned proposal passes every ADR 0115 proposal gate.

Classify `extended_exposure_tile_insufficient` when the pipeline passes but any
strength gate fails. Classify `extended_exposure_pipeline_invalid` before
interpreting strength if any identity, numerical, coverage, replay, resource,
or sealed-domain gate fails.

A sufficient result freezes this tile proposal and moves to the remaining
selector gate. An insufficient result closes uniform full-data exposure and
requires one sampling or optimization-schedule audit. No result directly
opens sealed test, gameplay, or a full policy/value trainer.

## Cluster Execution

- john2 owns the sole 200-epoch MLX origin.
- john3 owns source identity, trajectory checks, and the required selected
  checkpoint replay.
- john4 owns the oracle-other-stage mixed ceiling.
- john1 owns implementation, tests, reporting, and integrated hierarchy
  evaluation.

The nontraining hosts prepare their dependency-free work during the origin and
launch their dependent evaluations immediately after the selected checkpoint
arrives. Duplicate origins are prohibited.

## Maximum Compute

One 200-epoch full-cache origin, one cross-host replay, one mixed-stage ceiling,
one integrated hierarchy evaluation, focused and full tests, one report, and
documentation. No early stop, second seed, confirmation replica, epoch sweep,
curriculum, new data, teacher rollout, sealed test, gameplay, cloud, Modal, or
external compute.

## Result

The sole john2 origin completed all 200 epochs and selected epoch 197 by the
frozen train-only rule. Train target recall reached 99.80% with 96.45% exact
query recovery, proving that the model can nearly memorize the full training
cache. Validation recall nevertheless fell from ADR 0116's 70.59% to 67.75%,
with 42.53% exact queries.

The checkpoint replayed bit-identically on john3. Peak process RSS was
3.09 GiB with zero swaps. The john4 oracle-other-stage ceiling retained only
64.95% of validation targets and 83.75% of winners. The integrated proposal
retained 64.42% of targets and 83.75% of winners. Every integrity gate passed;
four strength gates failed.

The mechanical classification is
`extended_exposure_tile_insufficient`. Uniform epoch extension is closed.
ADR 0119 independently failed the target-mass sampling-mismatch gate, so the
only authorized successor is one optimizer-schedule treatment with all other
model, objective, data, seed, width, and evaluation variables frozen.

Combined scientific BLAKE3:
`3e01e3b0cc1d55f54f3ec880deb0459a3ef09609d6594714ce3cd78578f7e555`.
