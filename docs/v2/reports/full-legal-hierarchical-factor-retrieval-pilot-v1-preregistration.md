# Full-Legal Hierarchical Factor Retrieval Pilot V1 Preregistration

Date: 2026-06-16

Experiment ID: `full-legal-hierarchical-factor-retrieval-pilot-v1`

## Question

Can three separately trained conditional MLX rankers reproduce the
`16 / 32 / 8` hierarchical oracle closely enough to retain more than 98% of
the high-budget top-64 target and then rank the retained actions with less
than 0.15 mean R4800 regret?

## Frozen Treatment

- Exact ADR 0114 factor partition and conditional order.
- Draft, tile, and wildlife widths `16 / 32 / 8`.
- Champion-frontier inclusion and frontier-anchored width-64 selection.
- One 256-wide set ranker per stage.
- Smooth-L1 absolute-rank, scale-16 listwise, and balanced boundary losses.
- AdamW `3e-4`, weight decay `1e-4`.
- 20 draft epochs, 20 tile epochs, 10 wildlife epochs.
- Train-only checkpoint selection and one final open-validation evaluation.

The proposal diagnostic uses the frozen expected-rank selector inside the
learned proposal. The deployable result uses the sum of the three learned stage
scores and must pass independently.

## Cluster Allocation

Cache shards are built as ten divisible jobs across john1-john4. The three
different stage models then train concurrently on three Macs while the fourth
performs the independent cache/oracle audit and prepares integration. Finished
hosts are dynamically backfilled with cross-host replay and integration work.
Duplicate discovery training is prohibited.

## Decision Rule

The pipeline must pass every identity, coverage, numerical, replay, resource,
and sealed-domain gate.

The learned proposal and learned top-64 result must each exceed 98% target
recall and 98% R4800 winner retention. The proposal must average no more than
2,048 actions. The deployable top 64 must have mean R4800 regret below 0.15 and
meet the frozen phase and subset stability thresholds.

Classify invalid pipeline, insufficient proposal, insufficient selector, or
`hierarchical_factor_retrieval_sufficient` in that precedence order. Only the
sufficient outcome completes Phase 2 and authorizes Phase 3.
