# ADR 0126: Oracle-Proposal Complete-Action Selector Feasibility

Status: preregistered

Date: 2026-06-16

Experiment ID: `oracle-proposal-complete-action-selector-v1`

## Context

ADR 0119 proved two facts about the next selector stage:

- fixed aggregation of oracle draft, tile, and wildlife ranks retained only
  74.72% of validation target actions at width 64; and
- raw stage-score dispersion differed by 5.64x on train and 9.42x on
  validation.

A learned complete-action selector is therefore mandatory whenever a tile
checkpoint passes the proposal gate. ADR 0120 is still the sole active tile
origin and this experiment cannot alter, accelerate, stop, or interpret it.

ADR 0097 previously tested four complete-action factor-integration
architectures over every legal action. They reached only 29.39%-30.88% train
target recall. That result confounded representation sufficiency with a much
wider action set than the hierarchical proposal. The same immutable
pre-compression factors can now answer a narrower question: does restricting
the selector to the oracle hierarchical proposal make any existing
architecture sufficient?

## Frozen Data

Use only the open train and validation caches from:

- `complete-action-frontier-factor-integration-v1`; and
- `full-legal-hierarchical-factor-retrieval-pilot-v1`.

Their dataset-manifest BLAKE3 must match. Group order, action counts, and every
action hash must agree exactly before filtering.

For each group, retain:

1. every champion-frontier action; and
2. every nonfrontier action whose draft, conditional tile, and conditional
   wildlife factors all belong to the oracle top `16 / 32 / 8` sets.

Keep the original complete-action top-64 target, R4800 values, source flags,
screen ranks, action hashes, selected action, game phase, Nature Token
availability, and whether the selected action is an independent draft. If the
selected action is outside the oracle proposal, record index `-1`; never add
it silently.

The filtered cache is diagnostic. It uses oracle factor membership to isolate
selector feasibility and is not a deployable proposal.

## Frozen Arms

Reuse the exact ADR 0097 pre-compression factors and architectures:

- `wide-concat`;
- `screen-relative`;
- `factor-attention`; and
- `pairwise-gated`.

Each arm trains from scratch for 20 epochs with its original seed, AdamW,
learning rate `3e-4`, weight decay `1e-4`, balanced per-group target-membership
BCE, and width-64 champion-frontier-anchored evaluation.

Checkpoint selection uses train metrics only, ordered by target recall, exact
target-set recovery, then lower train loss. Validation is evaluated exactly
once after the train-selected checkpoint is frozen.

No architecture, feature, target, optimizer, seed, epoch, selector, or width
sweep is allowed.

## Gates

Every arm must preserve source identities, cover every filtered group and
action exactly once, remain finite, use less than 4 GiB peak process RSS,
perform zero process swaps, and keep sealed test, gameplay, teacher compute,
cloud, Modal, and external compute closed.

An arm is selector-feasible only when:

- train target recall is at least 95%;
- train exact target-set recovery is at least 50%;
- validation target recall is at least 90%;
- validation R4800 winner retention is at least 98%;
- validation mean retained R4800 regret is below 0.15; and
- every phase and Nature Token/independent-draft selector guardrail from
  ADR 0115 passes.

Select the smallest feasible architecture in this order:

1. `wide-concat`;
2. `pairwise-gated`;
3. `factor-attention`;
4. `screen-relative`.

Classify `oracle_proposal_selector_feasible` when at least one arm passes.
Classify `oracle_proposal_selector_representation_insufficient` when every
pipeline passes but no arm passes. Classify
`oracle_proposal_selector_pipeline_invalid` before interpreting strength when
any cache, alignment, coverage, numerical, resource, or closed-domain gate
fails.

This feasibility result cannot promote a selector. A passing tile proposal
still requires one separately preregistered final selector trained on that
frozen learned proposal.

## Cluster Execution

- john1 builds the train cache and trains `wide-concat`, then
  `screen-relative`;
- john3 builds the validation cache and trains `factor-attention`;
- john4 verifies source identity and trains `pairwise-gated`;
- john1 collects and mechanically classifies all four arms.

Train and validation cache construction are disjoint unique work. The four
model arms are distinct hypotheses, not replicas. john2 continues ADR 0120's
sole tile origin throughout.

## Maximum Compute

One filtered train cache, one filtered validation cache, four 20-epoch
architecture arms, one selected-arm cross-host replay only if an arm passes,
focused and full tests, one report, and documentation. No tile training,
teacher rollout, sealed test, gameplay, cloud, Modal, or external compute.
