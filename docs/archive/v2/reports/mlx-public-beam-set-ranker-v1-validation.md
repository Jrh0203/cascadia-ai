# MLX Public Beam Set Ranker v1 Validation

Status: rejected before sealed-test access.

The selected epoch-5 checkpoint improved the immediate-score baseline, but it
missed both frozen decision-fidelity gates.

| Metric | Result | Gate | Pass |
|---|---:|---:|---:|
| Centered advantage correlation | 0.7891 | >= 0.70 | Yes |
| Tie-aware top-value recall | 0.3516 | >= 0.40 | No |
| Mean top-action regret | 0.3730 | <= 0.35 | No |

Immediate-score argmax achieved 0.2969 recall and 0.4873 regret. Joint
candidate attention therefore learned useful decision signal, but not enough
to unlock the sealed test. No model was promoted and no gameplay ran.
