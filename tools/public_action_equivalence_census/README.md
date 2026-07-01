# Public Action Equivalence Census

This standalone Rust tool implements the S7 foundation from
`docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md`.

It does not treat the earlier R3 local-token multiset collision as a complete
afterstate equivalence. Instead, it:

- reconstructs every graded-oracle action against its authoritative game;
- applies the accepted exact R3 edit to obtain the full semantic public
  afterstate and exact semantic supply without cloning a 441-cell board;
- includes the ordered market-prelude and draft side-effect trace in the
  serving-safe key;
- authoritatively reconstructs every proposed collision;
- rejects same-semantic-state near matches with different paid-wipe or draft
  traces;
- verifies exact serialized-state and exact hidden-successor subclasses; and
- reports full-legal-set reduction before any serving integration.

The production corpus is sharded by `(split_row % shard_count) == shard_index`.
Train and validation are open; test and final are rejected.

```bash
cargo run --release --manifest-path tools/public_action_equivalence_census/Cargo.toml -- \
  adversarial \
  --output artifacts/experiments/s7-public-action-equivalence-foundation-v2/adversarial.json

cargo run --release --manifest-path tools/public_action_equivalence_census/Cargo.toml -- \
  duplicate-smoke \
  --dataset-root artifacts/datasets/complete-action-graded-oracle-v1-train \
  --output artifacts/experiments/s7-public-action-equivalence-foundation-v2/duplicate-smoke.json

cargo run --release --manifest-path tools/public_action_equivalence_census/Cargo.toml -- \
  census \
  --dataset-root artifacts/datasets/complete-action-graded-oracle-v1-train \
  --dataset-root artifacts/datasets/complete-action-graded-oracle-v1-validation \
  --shard-index 0 --shard-count 3 \
  --source-bundle-blake3 <64-hex-digest> \
  --output artifacts/experiments/s7-public-action-equivalence-foundation-v2/shards/shard-0.json
```

The aggregate command accepts every disjoint shard plus the adversarial report
and emits forward, reverse, and byte-order proof artifacts.

Protocol V2 supersedes the invalid V1 attempt. V1 was stopped before producing
an accepted shard after a review found that semantic collapses beyond exact
public identity were counted as `n - k` instead of `k - 1`, where `n` is the
serving-safe class size and `k` is its number of exact-public subclasses.
