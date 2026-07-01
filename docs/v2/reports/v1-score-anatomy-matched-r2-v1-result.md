# V1 Matched Score-Anatomy Result

- Experiment: `v1-score-anatomy-matched-r2-v1`
- Decision: ADR 0176
- Reproducibility repair: ADR 0178
- Classification: `v1_score_anatomy_not_promoted`
- Classification ID: `721b97e61b8156e45e90ff062923857e15b0677ee5d327e23836a888d788d740`

## Result

The component-anatomy treatment reproduced exactly across hosts but failed its
preregistered ranking and calibration gates. It is not promoted.

The scalar and component arms used the same 92-object Exact-R2 state, identical
142,283-parameter graph, initialization, examples, optimizer, and 3,000-step
schedule. Both primary/replay pairs produced byte-identical final model
tensors and identical validation prediction probes.

## Matched Comparison

| Metric | Scalar control | Component anatomy | Anatomy minus scalar | Gate |
|---|---:|---:|---:|---|
| Total MAE | 2.4871 | 2.4061 | -0.0811 | Pass |
| Total RMSE | 3.1894 | 3.0818 | -0.1077 | Supporting improvement |
| Total correlation | 0.4276 | 0.3387 | -0.0889 | Fail |
| Pairwise accuracy | 0.6565 | 0.6299 | -0.0266 | Fail |
| Pairwise log loss | 0.6645 | 0.6930 | +0.0285 | Fail |

Component supervision made the individual score channels substantially more
interpretable: mean component MAE fell from 4.0638 to 2.2749, and mean wildlife
component MAE fell from 6.2219 to 3.4039. That decomposition benefit did not
survive summation into the action-ranking signal. The treatment compressed or
misaligned total-score variation enough to reduce correlation and worsen
within-round choice calibration.

## Reproducibility Repair

The original classifier incorrectly included
`optimization.checkpoint_manifest_blake3` in the role-neutral equality
predicate. That manifest transitively included elapsed wall time. ADR 0178
removed only that runtime field from role-neutral comparison while retaining
all tensor, optimizer, step, example-count, probe, data, graph, and
authorization checks.

The repaired classifier was sealed in immutable bundle
`9f369c7a46a63a068b25f63aa6e46574d211f4a86dd36c5bbe6c33745cfbf8ab`.
It reproduced classification ID
`721b97e61b8156e45e90ff062923857e15b0677ee5d327e23836a888d788d740`
byte-for-byte from the frozen reports.

## Interpretation

KataGo-style score anatomy is not sufficient as a replacement objective for
this value head. Future use of score components should preserve a directly
trained scalar or distributional action objective and treat component heads as
auxiliary regularizers, diagnostics, or conditional features. No gameplay
qualification is authorized from this result.

## Artifacts

- Classification: `artifacts/experiments/v1-score-anatomy-matched-r2-v1/classification.json`
- Repair bundle: `artifacts/experiments/v1-score-anatomy-matched-r2-v1/repairs/9f369c7a46a63a068b25f63aa6e46574d211f4a86dd36c5bbe6c33745cfbf8ab`
- Primary and replay reports: `artifacts/experiments/v1-score-anatomy-matched-r2-v1/reports`
- Role-neutral parity decision: `docs/v2/decisions/0178-v1-score-anatomy-role-neutral-checkpoint-parity.md`
