# MLX Public Beam Value v1 Validation

Status: rejected before sealed-test access.

The frozen Apple GPU run completed 20 epochs in 85.51 seconds. It passed only
the centered-advantage correlation gate.

| Metric | Result | Gate | Pass |
|---|---:|---:|---:|
| Terminal MAE | 2.7682 | <= 1.00 | No |
| Raw value correlation | 0.5830 | >= 0.90 | No |
| Centered advantage correlation | 0.6730 | >= 0.65 | Yes |
| Exact top-action agreement | 0.1406 | >= 0.50 | No |
| Mean top-action regret | 0.7280 | <= 0.35 | No |

The sealed test artifact was collected in advance but not evaluated. No model
was promoted and no gameplay benchmark was run.

Decision-level validation diagnostics strengthen the interpretation. The two
R8 teacher halves agreed on the winner 54.69% of the time, and either half's
winner cost at most 0.0449 against their mean. Immediate-score argmax reached
0.2969 tie-aware top-value recall and 0.4873 regret, beating the rejected model
at 0.2266 recall and 0.7280 regret. ADR 0040 therefore tests joint candidate-set
attention with an immediate-score residual anchor.
