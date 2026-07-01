# ADR 0066: MLX Joint Return and Root-Ranking Fine-Tuning

Status: rejected on held-out validation on 2026-06-12. Conditional gameplay
seeds 34,699-34,702 remain unopened.

## Context

ADR 0065 proved that the qualified sparse NNUE can learn complete-rollout
score-to-go much more accurately: held-out RMSE fell 41.84% and
within-personal-turn residual Pearson improved by 0.0591. It still failed
before gameplay because root pairwise accuracy and selected-action top-one
both regressed slightly.

Absolute trajectory targets are dominated by state and turn offsets. Search
needs the much smaller differences among candidate afterstates from the same
decision. ADR 0065 already retained exact R600 root groups, including each
candidate's immediate score, rollout mean, uncertainty, sample count, and
selected bit. The next objective must train those within-decision differences
directly while retaining the trajectory signal.

## Decision

Fine-tune the same six value tensors of the qualified 11,231-512-64-1 sparse
NNUE from the unchanged qualified parent. Keep the policy head byte-identical.

Each epoch performs:

1. one deterministic shuffled pass over every ADR 0065 train trajectory with
   the existing four-point Huber score-to-go loss;
2. one deterministic shuffled pass over every complete train root group in
   batches of four decisions;
3. exact final-score assembly as `immediate score + predicted remaining`;
4. group-centered one-point Huber regression over candidate final scores;
5. selected-action listwise cross-entropy with coefficient `0.50`;
6. soft teacher listwise cross-entropy at temperature `1.0` with coefficient
   `0.25`.

The selected-action and teacher distributions are training labels only. No
root metadata beyond the sparse afterstate and exact immediate score becomes
an inference input.

Train with AdamW learning rate `3e-6`, zero weight decay, trajectory batch 512,
root group batch four, seed 20260621, at most 12 epochs, and validation
patience four. Select the checkpoint with the lowest exact-kernel validation
root objective, not trajectory loss.

## Frozen Protocol

- Parent:
  `artifacts/models/legacy-nnue-v4opp-mlx-v1/model.json`, BLAKE3
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`.
- Train: reuse only the immutable ADR 0065 train dataset at indices
  94,000-94,003, manifest BLAKE3
  `5a041c73c15075e38d3106a77b09b1a33e6597c4d8eee5eea38446490c282ec0`.
- Validation: collect two fresh exact-MLX R600 games at validation indices
  95,000-95,001 with trace modulus eight.
- Teacher: exact MLX K32/R600, sequential halving, LMR enabled, diverse
  prefilter enabled, no semantic candidate injection.
- Implementation smoke: the already-open R32 index-93,000 dataset may verify
  grouping, loss, checkpoint, and resume mechanics only.
- No external compute, augmentation, old validation reuse, warm start from the
  rejected ADR 0065 checkpoint, coefficient sweep, or hyperparameter sweep.

The selected model advances to gameplay only if:

- datasets, source, checkpoint, parent, and packaged artifact pass integrity
  checks;
- validation trajectory RMSE improves at least 2% over the parent;
- validation within-personal-turn residual Pearson improves at least 0.02;
- no personal-turn quartile RMSE regresses by more than 0.10 point;
- root scored-action pairwise accuracy improves by at least 0.5 percentage
  point;
- root selected-action top-one does not regress;
- root conditional mean regret does not regress;
- all outputs are finite and the parent remains unchanged.

Passing every offline gate authorizes:

- gameplay smoke seed 34,699;
- paired pilot seeds 34,700-34,702;
- parent versus derived exact-MLX K32/R600 with independent warmed services;
- runtime ceiling 240 seconds/game per arm;
- derived-minus-parent mean at least +0.50, derived mean at least 96.00,
  wildlife at least -0.50, habitat at least -0.50, Nature Tokens at least
  -1.00, zero fallback, and clean shutdown.

Passing the pilot authorizes only a separately preregistered disjoint
confirmation. Any failed offline or gameplay gate closes this exact joint
objective.

## Maximum Compute

One implementation smoke on already-open R32 evidence, two fresh R600
validation games, one local MLX run of at most 12 epochs, and conditionally
one gameplay smoke plus three paired pilot games. No new train collection,
test split, sweep, external compute, or promotion.

## Implementation Smoke

The already-open R32 index-93,000 artifact completed one Apple-GPU epoch on
2026-06-12:

- 32 trajectory batches and 20 complete root-group batches;
- 52 optimizer steps in 1.86 seconds;
- exact-kernel trajectory and root validation;
- atomic checkpoint selection and bounded retention;
- schema-2 derived artifact packaging;
- byte-identical parent identity before and after;
- clean process exit.

The smoke's root selection loss moved from `3.33476` to `3.32784`. Its
pairwise and regret movements are implementation diagnostics only and cannot
select the substantive model.

## Result

Rejected before gameplay.

The fresh validation collection completed both games with 119,753 trajectory
records and 4,621 root records. Its manifest BLAKE3 is
`10adbe29432e10a91380c5a43a4391fe9695d7884925d7fe83533c311ced8dcf`.
The single Apple-GPU run completed all 12 epochs and selected epoch 12, step
6,720, in 231.88 seconds.

Against the qualified parent on the fresh split:

- trajectory RMSE improved from `5.03061` to `2.88953`, a `42.56%`
  reduction;
- within-personal-turn residual Pearson improved from `0.46068` to `0.52172`,
  a gain of `0.06104`;
- all four personal-turn quartile RMSEs improved;
- root selection loss improved from `2.80536` to `2.76335`;
- selected-action top-one improved from `0.27500` to `0.29375`;
- root pairwise accuracy regressed from `0.70911` to `0.70573`, missing the
  required `+0.005` improvement;
- conditional mean regret worsened from `1.01325` to `1.28064`.

The joint loss successfully increased exact selected-action recall but made
the remaining mistakes more expensive and degraded broad pairwise ordering.
The exact objective is therefore closed. No gameplay seed was opened. The
candidate artifact remains at
`artifacts/models/exact-mlx-joint-return-ranking-v1` for reproducibility only
and is not promoted.

The complete generated report is
`artifacts/runs/exact-mlx-joint-return-ranking-v1/final-report.json`, BLAKE3
`240532422ff708dda0bdb12ee3a3dde96e84d3a9ade0bce15e6266aec6e1ac7e`.
