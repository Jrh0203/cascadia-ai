# R4 bounded-parent MLX sidecar

`r4-bounded-parent-mlx-exporter` is the Rust-authored cache boundary for
ADR 0156. It replays the open graded-oracle train and validation datasets,
binds every group to the accepted ADR 0150 R3 cache, constructs exact R2 and
radius-four `CSR4AM1`, and stores all twelve D6 views for Q1, Q2, and Q3.

The output is content addressed and ragged. It contains token kinds, explicit
relative-seat owners, active `i16` values, offsets, counts, and canonical
state/view hashes. It does not contain candidate actions, labels, hidden bag
order, future refills, test data, or gameplay data.

```bash
cargo run --release \
  --manifest-path tools/r4_adaptive_multires_census/Cargo.toml \
  --bin r4-bounded-parent-mlx-exporter -- \
  --train-dataset artifacts/datasets/complete-action-graded-oracle-v1-train \
  --validation-dataset artifacts/datasets/complete-action-graded-oracle-v1-validation \
  --r3-cache artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache/0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156 \
  --output-root artifacts/experiments/r4-bounded-quotient-mlx-comparison-v1/cache \
  --receipt artifacts/experiments/r4-bounded-quotient-mlx-comparison-v1/cache-receipt.json
```

For a bounded mechanical smoke, add `--max-groups-per-split 1`. Production
training must require a complete sidecar.
