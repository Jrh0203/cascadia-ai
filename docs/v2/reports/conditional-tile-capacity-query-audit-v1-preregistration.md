# Conditional Tile Capacity and Query Audit V1 Preregistration

Date: 2026-06-16

Experiment ID: `conditional-tile-capacity-query-audit-v1`

## Question

Does the unchanged target-only tile ranker fail locally, fail only as many
queries share parameters, or need explicit candidate-to-candidate interaction?

## Frozen Arms

- john1: full open-split error anatomy and input-sensitivity audit, no training.
- john2: unchanged ranker on the nested 16-query hard cohort, 2,000 updates.
- john3: unchanged ranker on the nested 256-query hard cohort, 4,000 updates.
- john4: two-block self-attention ranker on the identical 256-query cohort and
  update budget.

All learning arms use seed `2026061649`, balanced target-membership BCE,
AdamW `3e-4`, weight decay `1e-4`, and batch size 16. Validation is not
evaluated by a learning arm.

## Decision Rule

- Fail the pipeline before interpreting strength.
- If the 16-query baseline misses 99.5% recall or 95% exact recovery, classify
  local fit insufficient.
- If the 16-query arm passes and the 256-query baseline reaches 98% recall and
  90% exact recovery, classify full-data scale or optimization insufficient.
- If the medium baseline fails but attention reaches 98%/90% with at least
  +5 recall points and +10 exact points, classify relational representation
  insufficient.
- Otherwise classify shared capacity or optimization insufficient.

No result opens sealed test, gameplay, a full trainer, or a sweep.

## Compute

Run four distinct arms concurrently across john1-john4. Replicas are
prohibited. Finished hosts backfill integrity, reporting, tests, or the next
dependency-ready task.
