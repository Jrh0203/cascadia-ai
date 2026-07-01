# S4 Candidate-Relation Foundation V1 Preregistration

Date: 2026-06-17

Status: frozen before production execution

Experiment ID: `s4-candidate-relation-foundation-v1`

Decision: ADR 0151

## Question

What is the smallest observable candidate anchor set that preserves the
high-fidelity decision target and exposes enough exact relational structure
for a linear-memory S4 ranker?

## Inputs

- complete open train split: 560 groups, with the frozen R3 at-most-512 cohort;
- complete open validation split: 240 groups and 860,203 legal actions;
- R3 cache `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156`;
- ADR 0150 open-data verification
  `a056aceadb7f53c01dc87c8a39d95a7866bac6df93b050c45cc860de2b8b87ea`;
- exact action records and authoritative afterstate hashes.

No sealed test split, gameplay result, hidden bag order, future refill, future
action, or model prediction is an input.

## Exact Relation Definitions

`same_draft`

: Exact action bytes after zeroing tile coordinate, rotation, wildlife
  presence/destination, immediate score, and immediate score decomposition.
  Tile identity, wildlife choice, independent-draft mode, market slots,
  mulligan sequence, staged market, staged public supply, and nature-token
  state remain bound.

`same_frontier`

: Exact tile axial coordinate.

`same_tile_pose`

: Exact tile axial coordinate and rotation.

`same_wildlife_destination`

: Exact wildlife axial coordinate for actions that place wildlife. Actions
  without a wildlife placement do not form a synthetic shared class.

`same_sibling_plan`

: Exact `same_draft` key plus exact tile pose. This identifies alternatives
  that share the draft and tile placement but differ in wildlife destination
  or equivalent downstream detail.

`equivalent_afterstate`

: Exact authoritative R2 afterstate hash from the R3 cache.

## Anchor Sets

Rank candidates by the observable screen rank, breaking ties by exact action
hash. Evaluate the first 64, 128, and 256 candidates. When a group is narrower,
the anchor set is the complete group.

For every width report:

- winner retention;
- pairwise 95% R4800 confidence-set coverage;
- retained R4800 regret;
- labeled-target retention;
- per-relation and union query-to-anchor coverage;
- winner-to-anchor and confidence-set-to-anchor linkage;
- relation collisions and pair edges;
- union graph components, largest component, and isolated anchors;
- dense `K^2` pair-score cost; and
- inducing-point `2KM` pair-score cost for `M = 8, 16, 32`.

Report all metrics overall and by opening/early, middle, and late phase and by
action-width buckets `1..512`, `513..2048`, `2049..4096`, and `4097+`.

## Frozen Selection

Select 128 anchors if it passes every ADR 0151 gate. Otherwise select 256 if it
passes. If neither passes, classify
`fixed_anchor_context_insufficient` and design adaptive or all-candidate
inducing context.

If 128 passes, compare 128 versus 256 only as a serving-cost ablation in the
later S4 neural tournament. Do not use this census to tune arbitrary widths.

## Distributed Protocol

Run modulo-three shards:

| Host | Rows |
|---|---|
| john2 | `row % 3 == 0` |
| john3 | `row % 3 == 1` |
| john4 | `row % 3 == 2` |

Each shard processes both open splits. The merge must contain train rows
`0..559` and validation rows `0..239` exactly once. Merge once in forward
report order and once in reverse report order; require byte-identical output.

## Success Classifications

- `s4_anchor_128_authorized`: 128 passes every frozen gate.
- `s4_anchor_256_authorized`: 128 fails and 256 passes.
- `fixed_anchor_context_insufficient`: neither width passes.
- `relation_surface_too_sparse`: oracle retention passes but relation linkage
  fails.
- `foundation_invalid`: any source, proof, row coverage, action identity,
  afterstate identity, finite-value, or order-invariance check fails.

This foundation authorizes a separately preregistered neural comparison. It
does not establish score improvement or permit gameplay promotion.
