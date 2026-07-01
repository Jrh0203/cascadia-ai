# ADR 0039: MLX Public Beam Value v1

Status: rejected on validation on 2026-06-12. Sealed test and gameplay were not
opened.

## Context

ADR 0038 established that final-five W2/B16 terminal values averaged over
public redeterminizations are observable from public action afterstates. The
next question is whether a local Apple MLX model can recover enough of that
counterfactual signal to improve play.

## Frozen Data

All datasets use the exact ADR 0038 teacher: public-market-only W2 roots,
B16/W2 final-five focal continuation, two disjoint R8 batches, K8+H6+B8+M4
opponents, and public-state-hash domain-separated seeds.

- train: train split indices `41000-41031` (32 games, 512 groups);
- validation: validation split indices `41000-41007` (8 games, 128 groups);
- test: test split indices `41000-41007` (8 games, 128 groups);
- one atomic checksummed shard per game;
- the test dataset remains sealed until one checkpoint passes every validation
  gate.

## Frozen Model

`mlx-public-beam-value-v1` uses the existing observable action-afterstate
entity encoder: hidden width 96, four attention heads, two board blocks, one
market block, and one scalar score-to-go head. Predicted final value is current
base score plus the learned residual.

The loss is fixed before collection:

- uncertainty-weighted Huber loss on terminal score-to-go;
- equally weighted within-group centered Huber loss;
- 0.25-weight listwise cross-entropy on candidate values;
- AdamW, learning rate `1e-4`, weight decay `1e-4`;
- group batch size 8, at most 20 epochs, patience 5, seed `20260612`.

Checkpoint selection minimizes `centered MSE + 0.1 * terminal MSE`.

## Gates

Validation must achieve all of:

- terminal-value MAE <= 1.00;
- terminal-value correlation >= 0.90;
- centered-advantage correlation >= 0.65;
- exact top-action agreement >= 50%;
- mean top-action regret <= 0.35.

The untouched test must then achieve MAE <= 1.15, value correlation >= 0.85,
centered correlation >= 0.60, top agreement >= 45%, and regret <= 0.50.

Only then may gameplay run. The fixed pilot is ten paired four-seat blocks
against promoted strong on seeds `31000-31009`. Promotion requires at least
`+0.50` paired mean, no material wildlife or habitat collapse, and runtime
below ten seconds per game before a disjoint 50-game confirmation.

No architecture, loss weight, data size, threshold, candidate frontier,
teacher sample count, or gameplay cutoff may be tuned from validation or test
results.

## Implementation Gate

Before collection, the split-aware Rust/Python pipeline, shared encoder, scalar
model, resumable trainer, validation-gated sealed-test evaluator, promotion
packager, binary MLX service, and Rust final-five W2 policy all passed their
focused tests and strict lint. A real ADR 0038 batch completed an Apple GPU
forward pass, backward pass, and AdamW update with finite changed predictions.

## Result

The frozen local collection completed and independently validated:

- train: 32 games, 512 groups, 10,116 candidates;
- validation: 8 games, 128 groups, 2,561 candidates;
- sealed test: 8 games, 128 groups, 2,548 candidates.

All three manifests share source digest
`f423b2762e20b6259c9a94cf1338e85c2585a72e84433fd962440eaa48210596`
and executable digest
`183bd729dd7f2bea3eb190f4119d98f42eb34590ec51aae97abd70958c2ddd7c`.
Training ran all 20 fixed epochs on `Device(gpu, 0)` in 85.51 seconds. The
selected epoch-20 checkpoint produced:

| Validation metric | Result | Gate | Pass |
|---|---:|---:|---:|
| terminal MAE | 2.7682 | <= 1.00 | no |
| raw value correlation | 0.5830 | >= 0.90 | no |
| centered advantage correlation | 0.6730 | >= 0.65 | yes |
| exact top-action agreement | 0.1406 | >= 0.50 | no |
| mean top-action regret | 0.7280 | <= 0.35 | no |

Only centered correlation passed. The model learned useful broad relative
signal, but independent candidate scoring did not identify the decision-level
winner and absolute calibration remained far outside gate. The validation
protocol therefore denied sealed-test access. No test metrics, promoted model,
or gameplay result exist.

This rejects the v1 independent scalar afterstate scorer, not the public target:
ADR 0038 established a repeatable counterfactual label. A successor must model
the complete candidate set as a decision and optimize decision fidelity
directly rather than treating each candidate as an isolated value example.
