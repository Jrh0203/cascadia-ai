# Exact-Parent Candidate-Set Residual Validation

Date: 2026-06-12

## Protocol

- ADR: `docs/v2/decisions/0069-exact-parent-candidate-set-residual.md`
- Train: 64 immutable R600 games, indices 51,000-51,063, split `train`
- Validation: 16 fresh R600 games, indices 51,032-51,047, split `validation`
- Parent: exact Rust-order MLX legacy NNUE
- Model: hidden 192, eight heads, three board blocks, one market block
- Optimizer: AdamW, learning rate `5e-5`, weight decay `1e-4`
- Batch: eight complete decision groups
- Seed: 20260622
- Selector: validation distributional loss only
- Stop: epoch 9 after six non-improving validation epochs
- Device: `Device(gpu, 0)` on Apple M4

## Data Integrity

The train sidecar contains 5,120 groups and 491,520 candidates. Fresh
validation contains 1,280 groups, 122,859 candidates, and 38,457/38,457
aligned teacher estimates. Every public state replayed exactly, every compact
action reconstructed and JSON-hash matched, and every parent prior aligned by
group, index, count, and action hash. One validation group retained 75 legal
actions; 96 is the configured cap, not a fixed group width.

## Result

Epoch 3 at step 1,920 was selected. Training completed 5,760 optimizer steps
over nine epochs in 155.510 seconds.

| Metric | Exact parent | Selected | Delta | Gate |
|---|---:|---:|---:|---|
| Distributional loss | 1.528932 | 1.397396 | -0.131536 | pass |
| Selected top-one | 21.016% | 22.891% | +1.875 pp | fail |
| Selected top-five | 49.453% | 54.688% | +5.234 pp | pass |
| MRR | 0.347279 | 0.377889 | +0.030610 | fail |
| Scored pairwise accuracy | 70.918% | 70.162% | -0.756 pp | fail |
| Value-difference correlation | 0.588722 | 0.573128 | -0.015593 | fail |
| Conditional mean regret | 0.747493 | 0.737381 | -0.010112 | fail |
| Teacher-frontier coverage | 73.125% | 76.953% | +3.828 pp | pass |
| Train selected top-one | 20.840% | 23.535% | +2.695 pp | fail |

## Conclusion

Rejected before test or gameplay. Candidate-set context improved loss,
top-five recall, and teacher coverage, but did not preserve broad value order
or materially reduce costly mistakes. The weak train top-one gain establishes
representation underfit rather than validation-only overfit. ADR 0069 is
closed; no architecture, coefficient, seed, threshold, or validation retry is
authorized.

The next experiment must add information. The strongest bounded candidate is
the exact parent's 64-dimensional hidden candidate representation, which may
retain distinctions discarded by its scalar score.
