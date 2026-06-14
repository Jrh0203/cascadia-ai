# Exact-Parent Hidden-State Residual Validation

Date: 2026-06-12

## Protocol

- ADR: `docs/v2/decisions/0070-exact-parent-hidden-state-residual.md`
- Train: 64 immutable R600 games, indices 51,000-51,063, split `train`
- Validation: 16 fresh R600 games, indices 51,048-51,063, split `validation`
- Parent: exact Rust-order MLX legacy NNUE
- Model: hidden-state candidate width 128, residual width 256
- Optimizer: AdamW, learning rate `5e-5`, weight decay `1e-4`
- Batch: eight complete decision groups
- Seed: 20260623
- Selector: validation distributional loss only
- Stop: full 30-epoch budget
- Device: `Device(gpu, 0)` on Apple M4

## Data Integrity

The train hidden sidecar contains 5,120 groups and 491,520 candidates. Fresh
validation contains 1,280 groups and 122,880 candidates, with every group at
the frozen 96-action cap. All 38,245 teacher estimates aligned. Every public
state replayed exactly, every compact action reconstructed and JSON-hash
matched, and every 64-value hidden record aligned by source dataset, shard,
group, index, count, and action hash.

The exact hidden service used MLX GPU and shut down cleanly. Its returned
scalar is bit-identical to the qualified exact operation. The selected and
latest checkpoint pointers both reload at epoch 30, step 19,200.

## Result

Epoch 30 at step 19,200 was selected. Training took 157.688 seconds.

| Metric | Exact parent | Selected | Delta | Gate |
|---|---:|---:|---:|---|
| Distributional loss | 1.522383 | 1.417843 | -0.104541 | pass |
| Selected top-one | 21.641% | 21.719% | +0.078 pp | fail |
| Selected top-five | 46.484% | 47.188% | +0.703 pp | fail |
| MRR | 0.343423 | 0.345758 | +0.002335 | fail |
| Scored pairwise accuracy | 70.993% | 71.398% | +0.405 pp | fail |
| Value-difference correlation | 0.616800 | 0.617046 | +0.000246 | pass |
| Conditional mean regret | 0.763917 | 0.762672 | -0.001245 | fail |
| Teacher-frontier coverage | 73.750% | 73.906% | +0.156 pp | pass |
| Train selected top-one | 20.840% | 20.840% | 0.000 pp | fail |

## Artifacts

- Run report:
  `artifacts/runs/exact-parent-hidden-state-residual-v5/adr70-report.json`
  (`27d67b5c4e28262548d5f6ff9a078d9214ccf4d857cb26f229fc4ed9395426a5`)
- Train hidden manifest:
  `artifacts/datasets/canonical-action-exact-parent-hidden-v1-train/dataset.json`
  (`ae54f38de7b9ab9aea87809b7824acd8a6e224152a953ade2cfd5c7da0d74cd1`)
- Validation action manifest:
  `artifacts/datasets/canonical-action-parent-hidden-v5-validation-actions/dataset.json`
  (`8bcef176a73d4258e7e35ede2878dffe9d842ab9803b7cddccc5213f9efb5f5c`)
- Validation target manifest:
  `artifacts/datasets/canonical-action-parent-hidden-v5-validation-targets/dataset.json`
  (`600343721bf4a820e69051d7aadff5fc804d0a496cbaf65b052c17e4518aad7c`)
- Validation hidden manifest:
  `artifacts/datasets/canonical-action-exact-parent-hidden-v1-validation/dataset.json`
  (`6894eec076524fe4f134daade48167d83ea7b0ff9da3ba8eaebc7bb3c3282ef1`)

## Conclusion

Rejected before test or gameplay. Exact hidden-state access improved
distributional calibration and slightly improved broad ordering, but did not
improve selected-action identification or meaningful regret. Exact train
top-one was unchanged, so this is not a held-out generalization failure.

ADR 0070 closes the legacy-parent representation branch. Before committing to
a fresh V2 policy architecture, the next step is an identifiability audit of
the existing R600 evidence: selected-action margins, standard errors,
statistically distinguishable winners, parent-rank coverage, and phase-wise
noise. Test and gameplay domains remain sealed.
