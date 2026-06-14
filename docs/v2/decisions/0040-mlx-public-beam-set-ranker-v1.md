# ADR 0040: MLX Public Beam Set Ranker v1

Status: rejected on validation on 2026-06-12. Sealed test and gameplay were not
opened.

## Context

ADR 0038 proved that public-redetermined B16/W2 continuation values are
repeatable. ADR 0039 then rejected an independent scalar candidate scorer:
centered validation correlation reached 0.6730, but mean regret was 0.7280 and
tie-aware top-value recall was 0.2266. The immediate-score argmax was stronger
at 0.4873 regret and 0.2969 recall. On the same validation groups, the two
independent R8 teacher batches agreed on an exact winner 54.69% of the time and
each batch winner incurred at most 0.0449 regret against their mean. The
failure is therefore candidate-set decision modeling, not target instability.

## Frozen Data

Reuse the immutable ADR 0039 artifacts without recollection:

- train: `public-beam-value-v1-train-32`, 512 groups, 10,116 candidates;
- validation: `public-beam-value-v1-validation-8`, 128 groups, 2,561
  candidates;
- sealed test: `public-beam-value-v1-test-8`, 128 groups, 2,548 candidates.

The sealed test remains inaccessible until every validation gate below passes.
No ADR 0039 checkpoint may initialize the successor.

## Frozen Model

`mlx-public-beam-set-ranker-v1` first applies the existing public
action-afterstate encoder independently to every legal candidate. It then:

- projects each candidate to hidden width 96;
- applies two four-head masked self-attention blocks across the complete legal
  candidate set;
- predicts a bounded correction to the explicit immediate score;
- returns `immediate_score + 4 * correction` as the decision score.

The board encoder remains two blocks, the market encoder one block, and the
feed-forward multiplier three. The service must process one Rust request as
one candidate group, not as independent singleton groups.

## Frozen Objective

The loss is:

- centered Huber regression on within-group continuation advantage;
- plus 0.50 hard-top cross-entropy, uniform across exact target ties;
- plus 0.25 soft listwise cross-entropy at teacher temperature 0.50.

Training is AdamW with learning rate `1e-4`, weight decay `1e-4`, group batch
size 8, at most 20 epochs, patience 5, and seed `20260613`.

Checkpoint selection minimizes:

`mean regret + 0.25 * (1 - top-value recall) + 0.10 * centered MSE`.

No width, depth, scale, temperature, loss weight, optimizer value, epoch
budget, threshold, or seed may be changed from validation or test results.

## Gates

Validation must achieve all of:

- centered-advantage correlation >= 0.70;
- tie-aware top-value recall >= 0.40;
- mean top-action regret <= 0.35.

The untouched test must then achieve centered correlation >= 0.65,
top-value recall >= 0.35, and regret <= 0.45.

Only a full test pass may unlock the same ten-block gameplay pilot on seeds
`31000-31009` against promoted strong. Promotion still requires at least
`+0.50` paired mean, category integrity, and runtime below ten seconds per
game before a disjoint 50-game confirmation.

## Interpretation

A pass would show that the public terminal signal becomes learnable when the
network is given the decision as a set and an immediate-score residual anchor.
A validation failure closes neural architecture work on this fixed teacher
and candidate representation; the next research axis must change the state,
target, or search process rather than train another ranker on the same corpus.

## Result

Implementation completed with grouped binary serving, sealed-test evaluation,
promotion integrity, Make targets, and focused tests. The full Python suite
passed 49 tests, Ruff passed, and a real frozen-data Apple GPU forward,
backward, and AdamW update produced finite changed predictions.

Training stopped after epoch 10 for five non-improving epochs. The selected
epoch-5 checkpoint produced:

| Validation metric | Result | Gate | Pass |
|---|---:|---:|---:|
| centered advantage correlation | 0.7891 | >= 0.70 | yes |
| tie-aware top-value recall | 0.3516 | >= 0.40 | no |
| mean top-action regret | 0.3730 | <= 0.35 | no |

The set ranker materially improved the immediate-score baseline: regret fell
from 0.4873 to 0.3730 and top-value recall rose from 0.2969 to 0.3516. It still
missed both decision-fidelity gates. The evaluator denied sealed-test access,
and no model was promoted or used in gameplay.

This is the final neural architecture experiment on the ADR 0039 corpus.
Future work must change the observable state, target construction, or search
process.
