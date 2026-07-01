# O1 Ranking Afterstate Exporter

This Rust tool replays the open complete-action graded-oracle trajectories and
materializes the exact public state immediately after every candidate selected
by the ADR 0188 top-64 cohort.

The exported O1 input records contain:

- `PositionRecord::observable_afterstate` for the candidate;
- up to eleven preceding champion actions;
- the candidate action as age zero;
- no opponent target, score target, policy identity, hidden post-draft refill,
  hidden stack order, or hidden bag order.

The output is content-addressed and binds the source datasets, cohort cache,
exporter binary, source tree, action hashes, source indices, model-input bytes,
and replay checks.

```bash
cargo run --release \
  --manifest-path tools/o1_ranking_afterstate_exporter/Cargo.toml \
  -- \
  --train-dataset artifacts/datasets/complete-action-graded-oracle-v1-train \
  --validation-dataset artifacts/datasets/complete-action-graded-oracle-v1-validation \
  --cohort artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/cohort/CACHE_ID \
  --output-root artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/afterstates \
  --receipt artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/afterstate-receipt.json
```
