# ADR 0175: Relational Selected-Prefix Pointer Pilot

Status: completed; `p1_pointer_tile_stage_insufficient`

Date: 2026-06-16

Experiment: `p1-relational-selected-prefix-pointer-pilot-v1`

Protocol: `matched-mlx-selected-prefix-pointer-pilot-v1`

Research-plan item: P1

## Context

ADR 0114 proved that the exact hierarchy
`draft 16 -> tile 32 -> wildlife 8` can retain 99.18% of the open-validation
expected-rank target and every R4800 winner at 482.4 mean proposals. ADR 0115
failed to learn that hierarchy from flattened pointwise factors: conditional
tile recall was 66.57%, integrated proposal recall was 72.48%, and the tile
membership gradient opposed the calibrated rank objective at cosine
`-0.738910`.

ADR 0174 replaces flattened tile rows with exact selected-prefix pointers over
the accepted sparse R2 state. Its implementation calibration mapped all
860,203 validation actions with zero pointer, action-identity, or D6 failures.
Production authorization still requires byte-identical complete train and
validation audits on two distinct hosts per split.

This pilot asks the next question only: can the exact pointer representation
learn the unchanged hierarchical retrieval target?

## Hypothesis

A state encoded once by the accepted exact-R2 C0 parent can support all three
conditional retrieval stages:

```text
structured draft object
-> active-board frontier token plus rotation
-> none, selected-prefix new tile, or occupied-tile token
```

The pointer model should remove the tile-stage representation conflict without
changing labels, legal actions, hierarchy widths, optimization budget, or
checkpoint-selection policy.

## Frozen Model

- Parent encoder: exact C0 parent from ADR 0161.
- Checkpoint:
  `step-000003000-epoch-0000-batch-003000`.
- Model file BLAKE3:
  `eadcfbd5d0f02d642e7003431809b9ae8c41f0c3faf12c57d6da84a18acc5b89`.
- Parent tensor BLAKE3 under the ADR 0175 named-tensor convention:
  `51c54d58edd536c139e5ff3b92cefe85d45bdfe5177387a4affa904dce7f73cf`.
- Parent width: 64 with four attention heads.
- Parent parameters are frozen and verified after every epoch.
- Draft head consumes the unchanged observable draft and staged-public prefix.
- Tile head points to one exact active-board frontier token and one legal
  rotation.
- Wildlife head points to the no-placement sentinel, the selected-prefix new
  tile, or one exact occupied token.
- Historical descendant statistics and pointwise local-geometry rows are
  excluded.

The active board uses at most 121 exact sparse objects by contract. This is not
an 11-by-11 dense crop and does not revive the 441-cell representation.

## Frozen Data And Objective

- Immutable ADR 0115 factor labels and action maps.
- Immutable exact-R2/R3 parent cache.
- Open train: 560 groups and 2,135,111 complete actions.
- Open validation: 240 groups and 860,203 complete actions.
- No sealed test, gameplay, future refill, or hidden order.
- Objective: unchanged rank regression plus scale-16 listwise calibration plus
  target-boundary loss.
- Widths: draft 16, tile 32, wildlife 8.
- Epochs: draft 20, tile 20, wildlife 10.
- Batch sizes: draft 32, tile 32, wildlife 256.
- AdamW learning rate `3e-4`, weight decay `1e-4`.
- Deterministic per-stage seeds.
- Exact D6 augmentation during training; identity orientation during scoring.

## Selection And Evaluation

Each stage selects its checkpoint on open-train metrics only, ordered by:

1. target-factor recall;
2. exact-query fraction;
3. lower expected-rank mean absolute error.

Open validation is scored exactly once after stage selection. Validation may
not choose an epoch, seed, width, objective, or architecture.

Each selected stage checkpoint and final report is then collected by the
coordinator, fanned out to a distinct host, and replayed over complete train
and validation. The replay metrics must match exactly before integration.

The three selected stages are then integrated through the unchanged action
maps and champion-frontier anchor contract. Complete actions are scored by the
sum of their selected-prefix factor logits and retained through the unchanged
top-64 selector.

## Performance Contract

The frozen C0 parent is encoded once per `(group, D6 transform)` within an
epoch and memoized for all conditional queries. This is an exact reuse of a
frozen function, not an approximation. Reports must expose requested parents,
actual parent encodes, cache hits, elapsed time, peak memory, and complete
query/item coverage.

Serving benchmarks must separately measure:

- one parent encode;
- draft, tile, and wildlife pointer scoring;
- integrated all-action reconstruction and top-64 selection;
- P50, P90, P99, and maximum latency;
- maximum resident and MLX active memory.

## Production Gate

Production launch is fail-closed until the ADR 0174 classifier says:

`p1_relational_pointer_foundation_passed`

and authorizes:

`matched-mlx-selected-prefix-pointer-pilot`

The trainer verifies the classification envelope, both split identities, C0
checkpoint manifest, model bytes, exact parent tensor set, parameter shapes,
and frozen parent hash.

## Success

All are mandatory on open validation:

- integrated proposal target recall greater than 98%;
- integrated R4800 winner retention greater than 98%;
- mean proposal count at most 1,024;
- target mean proposal count of 512 or less is preferred;
- top-64 confidence-set coverage at least 99%;
- mean retained R4800 regret below 0.15;
- every stage scores every query and item exactly once with finite logits;
- every selected stage replays exactly on a distinct host;
- no early, middle, late, Nature Token, independent-draft, or action-family
  guardrail failure;
- no parent mutation, target drift, action-map drift, or information leak.

Stage metrics are diagnostics, not substitutes for the integrated gates.

## Failure And Pivot

If exact pointer tile recall remains below 90%, P1 stops before gameplay and
the failure is classified:

1. query-state insufficiency;
2. pointer interaction/capacity insufficiency;
3. objective interference despite the representation change; or
4. target non-identifiability under public observables.

Only one preregistered forensic pass may distinguish these causes. Width,
objective, and label changes require a new ADR. No post-hoc dense-441 fallback,
coordinate MLP fallback, target widening, anchor inflation, or validation-based
checkpoint choice is allowed.

If offline P1 passes, complete-action rescoring and equal-budget gameplay are
separate successor experiments. This ADR cannot claim progress toward 100
points from offline recall alone.

## Result

Production completed on 2026-06-17. All three selected stage checkpoints
reproduced exactly on distinct hosts and every query and item was scored once
with finite logits. The frozen parent remained unchanged.

The tile stage reached only 48.51% validation target-factor recall, below the
90% failure threshold. Integrated learned top-64 target recall was 9.82%,
R4800 winner retention was 48.75%, confidence-set coverage was 72.50%, and
retained regret was 0.1945. The experiment is terminally classified
`p1_pointer_tile_stage_insufficient`; no gameplay or sealed data was opened.

See `reports/p1-relational-selected-prefix-pointer-pilot-v1-result.md`.
