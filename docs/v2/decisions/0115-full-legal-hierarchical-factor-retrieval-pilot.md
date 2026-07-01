# ADR 0115: Full-Legal Hierarchical Factor Retrieval Pilot

Status: complete; hierarchical proposal insufficient

Date: 2026-06-16

Experiment ID: `full-legal-hierarchical-factor-retrieval-pilot-v1`

## Context

ADR 0114 proved that exact prefix-conditional retrieval with widths
`16 / 32 / 8` retains 99.18% of the validation expected-rank target, 95% of
complete target sets, and every validation R4800 winner while reducing the
mean candidate set to 482.4 actions. The independent-factor control lost 4.52
recall points, so draft-conditioned tile retrieval is structurally necessary.

The oracle did not prove learnability. It also used expected-rank values for
the final top-64 selection and therefore cannot be deployed. This pilot tests
one learned hierarchy without changing the teacher, data, action factorization,
retrieval widths, frontier anchor, or open split boundary.

## Frozen Evidence

- ADR 0114 combined report BLAKE3:
  `0de5dedb15d1608068348bb2b2dd2d47f8b3ec27a27e3b7c4418379dea89e700`.
- ADR 0114 source bundle BLAKE3:
  `71e4ea6992621e014aa0bd7814e9f88ab5631a0a9249b33b10aadaaa69a1a0e6`.
- Complete-action train and validation datasets and scale-16 expected-rank
  caches are unchanged.
- The exact factor partition remains draft/premove `0:34 + 45:128`, tile
  placement `34:42`, and wildlife placement `42:45`.
- Champion-frontier actions remain unconditionally retained.
- Retrieval widths remain exactly `16 / 32 / 8`.

Sealed test, gameplay, new teacher compute, cloud, Modal, and external compute
remain closed.

## Shared Factor Cache

Before training, export one immutable cache shard per source dataset shard.
Cache construction is divisible across john1-john4 and contains only public
observables, expected-rank supervision already frozen by ADR 0101, factor
identities, deterministic prefix membership, and evaluation metadata.

Each factor item receives the minimum finite expected rank among its
descendant complete actions. The deterministic retrieval target is the first
`min(width, finite items)` factors ordered by expected rank and exact factor
bytes. Cache construction must prove:

- the three factor identities remain bijective with every complete action;
- staged market and tile-local features claimed to be prefix-invariant are
  byte-identical within that prefix;
- all train and validation groups and actions are represented exactly once;
- candidate-to-factor mappings are complete for every eligible nonfrontier
  action; and
- independently recomputed cache identities and ADR 0114 oracle metrics agree.

## Stage Features

All models receive the same lossless public parent state: four boards and
masks, market and mask, global features, and public supply.

The stage-specific observable features are:

1. **Draft/premove**
   - exact 117-value draft/premove factor;
   - staged market, mask, and public supply;
   - minimum, mean, and maximum public prior and immediate-consequence values
     over descendant actions; and
   - descendant count.
2. **Conditional tile**
   - query context: exact draft/premove factor and staged public state;
   - item: exact tile factor, the six rotation-canonical tile-neighbor
     relations, descendant prior/consequence statistics, and descendant count.
3. **Conditional wildlife**
   - query context: draft/premove, staged public state, tile factor, and the
     six tile-neighbor relations;
   - item: exact wildlife factor, the target plus six rotation-canonical
     wildlife relations, and the action's public priors and immediate
     consequences.

No R600, R1200, R4800, expected-rank, selected-action, or future-state value
may enter a model feature.

## Frozen Model

Train three separate MLX set rankers, one per stage. Each ranker uses:

- a dense public-state encoder;
- separate query-context and item encoders;
- multiplicative and absolute-difference state/query/item interactions;
- masked item mean and maximum context; and
- one scalar score per factor item.

Each stage uses hidden width 256, AdamW, learning rate `3e-4`, weight decay
`1e-4`, deterministic stage seeds, and no warm start. Draft and tile train for
20 epochs. Wildlife trains for 10 epochs because the width-eight budget
retains every observed prefix of at most five wildlife factors; its learned
score is still required for final complete-action ranking.

The per-query loss is the sum of:

- smooth-L1 regression to `-log1p(minimum expected rank)`;
- scale-16 listwise expected-rank cross entropy; and
- class-balanced top-width membership BCE when a query has both positive and
  negative items.

Queries, not actions, receive equal loss weight. A stage checkpoint is selected
only by frozen train metrics: target-factor recall, then exact query recovery,
then rank regression error. Validation is evaluated once after selection.

## Learned Retrieval And Selection

At inference:

1. retain the learned top 16 draft factors;
2. for each retained draft retain the learned top 32 tile factors;
3. for each retained draft+tile retain the learned top eight wildlife factors;
4. add every champion-frontier action; and
5. apply the unchanged frontier-anchored width-64 selector using the sum of the
   three calibrated stage scores for each complete action.

Exact factor bytes break stage ties. Action hashes break final ties.

The report also applies the frozen expected-rank selector inside the learned
proposal. This diagnostic isolates retrieval from learned final ranking and
cannot by itself pass Phase 2.

## Gates

Pipeline gates require:

- exact cache coverage, factor bijection, prefix invariants, and mapping
  integrity;
- matching source, data, cache, model, and weight identities;
- finite losses, scores, gradients, parameters, and optimizer state;
- deterministic selected-checkpoint replay on another host for every stage;
- less than 4 GiB peak process RSS, zero process swaps, and no attributable
  positive system-swap growth;
- sealed test, gameplay, new teacher compute, cloud, and external compute
  remain unused.

The learned proposal gate requires on both train and validation:

- expected-rank target-positive recall greater than 98%;
- R4800 winner retention greater than 98%;
- mean proposal count at most 2,048; and
- validation recall at least 97% in every game phase and at least 95% in the
  Nature Token and independent-draft subsets when each contains 20 groups.

The deployable top-64 gate requires:

- expected-rank target-positive recall greater than 98%;
- mean retained R4800 regret below 0.15;
- R4800 winner recall greater than 98%;
- at least 97% winner recall and regret below 0.20 in every game phase; and
- at least 95% winner recall and regret below 0.25 in the Nature Token and
  independent-draft subsets when each contains 20 groups.

## Mechanical Classification

1. `hierarchical_retrieval_pipeline_invalid`
   - any identity, coverage, integrity, replay, resource, numerical, or sealed
     gate fails.
2. `hierarchical_proposal_insufficient`
   - the pipeline passes but the learned proposal misses a proposal gate.
3. `hierarchical_selector_insufficient`
   - the learned proposal passes but the learned top-64 selector misses a
     deployable gate.
4. `hierarchical_factor_retrieval_sufficient`
   - every pipeline, proposal, and deployable top-64 gate passes.

Only the sufficient result completes Phase 2 and authorizes the Phase 3 policy
and value pilot. A selector-only failure authorizes one ranking treatment over
the frozen learned proposal. A proposal failure requires a mechanistic audit
before another learned hierarchy.

## Cluster Execution

Wave 1 dynamically schedules the ten source-shard cache exports across all four
Macs, followed by cache combination and an independent cache audit.

Wave 2 assigns one distinct model to each of three Macs: draft, conditional
tile, and conditional wildlife. The fourth Mac runs the independent cache and
oracle-reconstruction audit plus integration tooling. These are three
different learning problems, not duplicate seeds.

As each model finishes, its host is backfilled with a cross-host replay or
dependency-ready integration task. There is no fixed three-model barrier for
work that can start earlier. Replication beyond the required selected-weight
replay is forbidden unless a preregistered failure diagnosis requires it.

Every campaign report includes per-host work, useful CPU and MLX occupancy,
idle time while compatible work was queued, duplicate-compute fraction,
campaign makespan, and decisions completed per elapsed hour.

## Maximum Compute

Ten cache-shard exports, one combine, one independent cache audit, three model
origins, three selected-weight cross-host replays, one train/validation
integration evaluation, focused and full tests, one report, and documentation.
No architecture, width, objective, seed, learning-rate, epoch, or feature
sweep.

## Result

Every pipeline gate passed. The immutable factor caches reproduced ADR 0114,
all three origins remained finite below 4 GiB with zero process swaps, and
draft, tile, and wildlife selected checkpoints replayed bit-identically across
hosts. Sealed test, gameplay, new teacher compute, cloud, and external compute
remained closed.

Stage validation factor recall was:

- draft: 92.84%;
- conditional tile: 66.57%; and
- conditional wildlife: 100.00%.

The integrated learned proposal retained 72.48% of validation target actions,
92.08% of validation R4800 winners, and 1,061.7 actions on average. The learned
top 64 retained 18.14% of validation targets and 58.75% of winners, with
0.140092 mean regret. Proposal and selector gates both failed.

The mechanical classification is `hierarchical_proposal_insufficient`.
Machine-readable combined scientific BLAKE3:
`7dd039771e9e638ce566e69ac4cbf8a9079ef8dee3a46ca04e63c4d5b81526bf`.

The required mechanistic audit found zero exact input-label collisions. On the
eight widest supervised tile queries, the top-32 boundary gradient opposed the
combined regression/listwise gradient at mean cosine `-0.738910`; its norm was
`24.3817` versus `28.0464` for the combined auxiliary pressure. An oracle
reranker over the union of learned and screen-prior top 32 reached only 78.29%
validation tile recall, ruling out a simple blend.

ADR 0116 is authorized as the one mechanistically selected successor: train the
unchanged tile model from scratch using only balanced top-32 membership BCE.
