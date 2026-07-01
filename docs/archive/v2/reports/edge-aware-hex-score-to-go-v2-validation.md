# Edge-Aware Hex Score-To-Go Validation

ADR 0073 tested whether exact hex adjacency and oriented matching-terrain
relations improve H6 score-to-go discrimination over the frozen generic
entity-set encoder.

## Protocol

- immutable training corpus: 256 H6 games, 20,480 positions;
- fresh validation: 64 games, indices 64,000-64,063, 5,120 positions;
- graph model: width 96, four edge-aware message-passing blocks;
- training: one local Apple-GPU run, seed 20260624, at most 30 epochs;
- selected checkpoint: epoch 15, step 3,840;
- no warm start, architecture sweep, retry, test access, or gameplay.

## Result

| Metric | Set baseline | Hex graph | Delta | Required | Result |
|---|---:|---:|---:|---:|---|
| Final correlation | 0.3933 | 0.3417 | -0.0516 | >=0.50 and +0.05 | fail |
| Final MAE | 2.5415 | 2.7982 | +0.2567 | <=3.00 and <=+0.10 | fail |
| Pairwise accuracy | 64.7406% | 65.3890% | +0.6484 pp | +3.00 pp | fail |
| Pairwise log loss | 0.7628 | 0.7296 | -0.0331 | improve | pass |
| Selection metric | 1.0169 | 1.0095 | -0.0075 | minimize | descriptive |

Wildlife MAE remained inside every half-point regression guardrail:

| Component | Set baseline | Hex graph | Delta |
|---|---:|---:|---:|
| Bear | 3.7421 | 3.8378 | +0.0957 |
| Elk | 3.2778 | 3.2004 | -0.0774 |
| Salmon | 3.4756 | 3.6676 | +0.1920 |
| Hawk | 2.9696 | 2.8094 | -0.1602 |
| Fox | 2.9433 | 2.9319 | -0.0114 |

Training and inference remained finite. Checksummed dataset, source,
checkpoint, and target-identity validation passed. Warmed batch-256 inference
on `Device(gpu, 0)` measured 88.70 milliseconds P90 total, 0.346 milliseconds
per position, and approximately 2,908 positions per second.

## Decision

Rejected on validation. Exact local geometry improved soft pairwise log loss,
but regressed final-score correlation and MAE and added only 0.648 percentage
point of pairwise accuracy. The sealed test dataset was not collected and no
gameplay domain was opened.

The result closes geometry-only work on the existing single-trajectory H6
labels. The next experiment must improve the target signal with repeated
counterfactual public-information futures and explicit shared-supply and
opponent-demand context.

## Artifacts

- training report:
  `artifacts/runs/edge-aware-hex-score-to-go-v2/final-report.json`;
- selected checkpoint:
  `artifacts/runs/edge-aware-hex-score-to-go-v2/checkpoints/step-000003840-epoch-0015-batch-000000`;
- inference report:
  `docs/v2/reports/edge-aware-hex-score-to-go-v2-inference.json`;
- validation dataset:
  `artifacts/datasets/score-to-go-hexgraph-v2-validation/dataset.json`.
