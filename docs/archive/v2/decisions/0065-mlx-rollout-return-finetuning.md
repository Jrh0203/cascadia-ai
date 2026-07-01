# ADR 0065: MLX Rollout-Return Fine-Tuning

Status: preregistered on 2026-06-12. No listed substantive seed or index has
been opened.

## Context

ADRs 0061-0063 closed rollout resampling, generic root widening, and exact H6
candidate injection. The qualified exact-MLX teacher still averages in the
95-96 range, but its complete-game rollouts already produce the evidence
needed to improve the policy used inside those rollouts.

Each R600 root search simulates complete games. On every future turn for the
focal player, the qualified sparse NNUE selects one afterstate, and the
simulation eventually yields an exact terminal base score. The selected
afterstate can therefore be labeled directly with:

`terminal base score - exact afterstate base score`.

This avoids winner imitation, absolute state offsets across unrelated
decisions, and an additional simulation bill. It trains the same value
quantity consumed by the rollout policy.

## Decision

1. Add optional trace collection to the evaluator-independent batched rollout
   engine. Normal gameplay remains allocation-free when tracing is disabled.
2. Retain a rollout trace only when its deterministic rollout seed modulo
   eight is zero.
3. Record the root afterstate and every subsequently selected focal-player
   afterstate, preserving sparse feature order and multiplicity.
4. At terminal state, label every retained afterstate with exact remaining
   base score.
5. Also retain every root candidate's R600 mean, standard deviation, sample
   count, immediate score, sparse afterstate, and selected bit for held-out
   ranking diagnostics.
6. Store train and validation evidence in versioned, checksummed, resumable,
   one-game shards with exact source, executable, model, environment, split,
   index, and sampling provenance.
7. Fine-tune all value tensors of the qualified 11,231-512-64-1 model in MLX,
   initialized byte-exactly from the qualified artifact. Keep the unused
   policy head unchanged.
8. Train against terminal score-to-go with Huber loss at a four-point
   transition, AdamW learning rate `3e-6`, zero weight decay, batch 512, seed
   20260620, at most 12 epochs, and validation patience four. Select only the
   lowest validation loss.
9. Package the selected checkpoint as a derived sparse-NNUE artifact with
   immutable parent, dataset, optimizer, checkpoint, and file checksums.

The exact Rust-order Metal inference path remains unchanged and serves the
derived tensors after training.

## Frozen Protocol

- Implementation smoke: one train game at index 93,000, R32, trace modulus
  eight. It may verify wiring only and may not select a model.
- Train: split `train`, indices 94,000-94,003, four R600 exact-MLX games.
- Validation: split `validation`, indices 94,000-94,001, two R600 exact-MLX
  games.
- Teacher: exact MLX K32/R600, sequential halving, LMR enabled, diverse
  prefilter enabled, no semantic candidate injection.
- Parent model:
  `artifacts/models/legacy-nnue-v4opp-mlx-v1/model.json`, BLAKE3
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`.
- Trace sampling: unsigned rollout seed modulo eight equals zero.
- Target: terminal base score minus exact afterstate base score.
- No habitat bonuses, no external compute, no augmentation, no replay of old
  caches, no hyperparameter sweep, and no warm start other than the qualified
  parent.

The selected model advances to gameplay only if:

- both datasets validate byte-for-byte and contain at least 100,000 train and
  40,000 validation trajectory records;
- every sparse index is below 11,231 and duplicate indices are preserved;
- selected validation RMSE improves at least 2% over the parent;
- selected validation within-personal-turn residual Pearson correlation
  improves at least 0.02;
- no personal-turn quartile RMSE regresses by more than 0.10 point;
- root scored-action pairwise accuracy improves by at least 0.5 percentage
  point;
- root selected-action top-one does not regress;
- root conditional mean regret does not regress;
- all outputs are finite and the parent artifact remains unchanged.

If all validation gates pass, authorize:

- smoke seed 34,499;
- pilot seeds 34,500-34,502;
- parent versus derived exact-MLX K32/R600, independent warmed services;
- runtime ceiling 240 seconds/game per arm;
- advancement gates: derived-minus-parent mean at least +0.50, derived mean at
  least 96.00, wildlife at least -0.50, habitat at least -0.50, Nature Tokens
  at least -1.00, zero fallback, and clean shutdown.

Passing the pilot authorizes only a separately preregistered disjoint
confirmation. Any failed validation or gameplay gate closes this exact
rollout-return fine-tuning recipe.

## Maximum Compute

One R32 implementation smoke, four R600 train games, two R600 validation
games, one MLX run of at most 12 epochs, and conditionally one gameplay smoke
plus three paired pilot games. No test split, sweep, external compute, or
promotion.

## Implementation-Smoke Clarification

The authorized R32 implementation smoke at train index 93,000 completed before
any substantive index was opened. The qualified parent scored raw trajectory
Pearson `0.99071`; turn number dominates score-to-go so strongly that the
original raw-Pearson `+0.02` gate was mathematically impossible and did not
measure within-phase value quality. The frozen Pearson gate above therefore
uses pooled within-personal-turn residuals: prediction and target are each
demeaned separately for personal turns 1 through 20 before covariance is
pooled. Raw Pearson remains reported as a diagnostic. No training,
validation, gameplay, or other substantive ADR65 seed had been opened when
this metric definition was corrected.

## Result

Rejected on held-out validation on 2026-06-12. No conditional gameplay seed
was opened.

The frozen train split completed all four games with 245,603 trajectory
records and 9,650 root records. Its manifest BLAKE3 is
`5a041c73c15075e38d3106a77b09b1a33e6597c4d8eee5eea38446490c282ec0`.
The frozen validation split completed both games with 121,246 trajectory
records and 4,844 root records. Its manifest BLAKE3 is
`ee98a051423d4448173a8479ddbb2ff7ff614d9b358a82cc95549cca4d33b6e3`.
Both datasets passed the independent Rust validator.

The single registered MLX run stopped after epoch 11 at patience four and
selected epoch 7, step 3,360. Against the exact parent on validation:

- trajectory RMSE improved from `5.14326` to `2.99122`, a `41.84%`
  reduction;
- within-personal-turn residual Pearson improved from `0.41853` to `0.47761`,
  a gain of `0.05907`;
- every turn-quartile RMSE improved by more than one point;
- conditional mean root regret improved from `1.05553` to `1.01389`;
- root pairwise accuracy regressed from `0.71834` to `0.71682`, missing the
  required `+0.005` improvement;
- selected-action top-one regressed from `0.28125` to `0.27500`, violating
  non-regression.

The model learned terminal score-to-go substantially better but did not
preserve root action ordering. The exact recipe is therefore closed before
gameplay. The schema-2 candidate artifact is retained at
`artifacts/models/exact-mlx-rollout-return-v1` for reproducibility only and is
not promoted. The complete generated report is
`artifacts/runs/exact-mlx-rollout-return-v1/final-report.json`, BLAKE3
`7cb3e98a3055869d8f44c4906ddb45ce6a290322fc9eb4b8065265a242fa9943`.
