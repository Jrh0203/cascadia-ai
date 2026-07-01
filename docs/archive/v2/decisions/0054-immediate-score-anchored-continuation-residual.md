# ADR 0054: Immediate-Score Anchored Continuation Residual

Status: rejected on validation on 2026-06-12. No test, promotion, or gameplay
domain was opened.

## Context

ADR 0053 retained every K32 teacher estimate but rejected an unanchored
pairwise scalar. Its best validation checkpoint reached 0.4443
value-difference correlation yet only 13.75% selected top-one and 71.41%
teacher-frontier coverage.

The exact immediate-score baseline was materially better at the top:

- top one 20.63% versus 13.75%;
- MRR 0.2923 versus 0.2692;
- teacher-frontier coverage 76.48% versus 71.41%.

A post-result reciprocal-rank blend reached 22.81% top-one but only 49.06%
top-five and 0.3587 MRR. It is diagnostic only and cannot be promoted. The
result shows that preserving exact short-horizon score helps, but a post-hoc
coefficient does not recover enough continuation structure.

The teacher's rollout mean is already a point-scale final-score estimate.
Learning an arbitrary ranking logit discards that decomposition. A more honest
target is:

`predicted final score = exact immediate score + learned continuation residual`.

## Decision

Add architecture `shared-state-action-score-residual-v3`:

1. Reuse the shared-state board, market, opponent, global, and explicit-action
   encoder.
2. Interpret its scalar output as a normalized continuation residual.
3. Return `immediate_score + 100 * residual` as the only training and
   inference score.
4. Zero-initialize the final residual head so the untouched model is exactly
   the immediate-score baseline rather than random noise.
5. Train only teacher-scored actions with confidence-weighted point regression:
   - point error divided by ten;
   - Huber transition at one normalized unit, or ten points;
   - confidence `1 / (1 + standard_error^2)`;
   - standard error from rollout standard deviation and sample count.
6. Add selected-action listwise cross-entropy across all 96 retained actions,
   using point logits divided by temperature five and coefficient 0.25.
7. Never expose rollout means, samples, uncertainty, source flags, or selected
   bits as inference inputs.

The fixed immediate term cannot be learned away inside score assembly. The
network can still predict a negative residual correction when an apparently
strong immediate move damages future score.

## Frozen Experiment

- Train dataset: reuse immutable ADR 0053 train dataset
  `imitation-targets-5d855c4ef1ee2edb`, 64 games at train indices
  51,000-51,063.
- Validation dataset: collect 16 fresh validation games at indices
  51,016-51,031.
- Test and gameplay domains: unopened and unauthorized.
- Teacher and candidates: unchanged deterministic K32/R600/LMR teacher,
  96-action full-frontier retention, immediate top 16.
- Model: fresh `shared-state-action-score-residual-v3`, hidden 96, four heads,
  two board blocks, one market block.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Training: batch 16, seed 20260617, at most 20 epochs, validation patience
  five.
- Checkpoint selection: anchored validation loss only.
- Warm start, augmentation, coefficient search, and post-hoc score blending:
  prohibited.

The selected checkpoint advances only if every gate passes:

- exact dataset, checkpoint, source, and resume integrity;
- 100% validation teacher-estimate alignment;
- anchored loss below untouched zero-residual initialization;
- selected top one at least 23% and at least two points above initialization;
- selected top five at least 50% and at least eight points above
  initialization;
- selected MRR at least 0.36 and at least 0.06 above initialization;
- predicted teacher-frontier coverage at least 80%;
- scored pairwise accuracy at least 70%;
- scored value-difference correlation at least 0.45;
- conditional mean regret at most 1.0 point.

Passing authorizes only a separately preregistered fresh test collection. A
failure rejects this exact point-residual objective before test access,
promotion, or gameplay.

## Maximum Compute

One local 16-game R600 validation collection and one Apple-GPU run of at most
20 epochs. No new train collection, external compute, warm start,
hyperparameter sweep, validation replay, threshold change, test collection,
promotion, or gameplay is authorized.

## Implementation Evidence Before Validation

- Exact tests prove zero residual returns immediate score in points.
- Loss tests prove point-accurate teacher predictions beat reversed
  predictions.
- A 140-shard paired-loader regression and completed-resume bookkeeping tests
  pass.
- A disposable one-game R2 smoke completed one Apple-GPU epoch and resumed
  into a second with the exact optimizer cursor.
- Cumulative runtime advanced from 0.185 to 0.380 seconds across resume.

These checks verify mechanics only and do not alter the frozen validation
protocol or gates.

## Result

The fresh validation collection completed all 16 games with 1,280 groups,
122,880 retained actions, and 38,705/38,705 aligned R600 estimates. The Apple
GPU run completed the frozen 20-epoch budget in 133.052 seconds. Anchored loss
fell from 4.984072 to 0.984838 at the selected epoch-17 checkpoint, but every
decision-quality gate failed:

- selected top one: 17.19% from an 18.91% initialization;
- selected top five: 43.52% from 36.48%;
- MRR: 0.3071 from 0.2896;
- teacher-frontier coverage: 76.64%;
- scored pairwise accuracy: 66.28%;
- value-difference correlation: 0.3805 from 0.5675;
- conditional mean regret: 1.1573 points.

The failure is target-semantic, not a checkpoint or runtime defect. Only
0.456% of continuation-residual variance on fresh validation was within
decision groups; the rest was overwhelmingly an action-independent state
offset. The network learned that easy absolute workload and blurred the small
within-state differences required for action selection.

ADR 0054 is rejected. Absolute remaining-score regression is closed. The full
result is in
`docs/v2/reports/canonical-action-score-residual-v3-validation.md`.

A disposable development-only centered-advantage probe then removed the
measured group offset on this already-open split. It improved teacher-frontier
coverage to 83.59% and value-difference correlation to 0.5459, but best
selected top-one still fell to 17.50% and regret rose to 1.1874. No fresh
domain was opened, and the unused implementation was removed. Target
centering alone is therefore not worth a substantive successor run.
