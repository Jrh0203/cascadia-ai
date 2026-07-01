# ADR 0053: Full-Frontier MCE Distributional Imitation

Status: rejected on validation on 2026-06-12. No test, promotion, or gameplay
domain was opened.

## Context

ADR 0049 proved that a shared-state MLX ranker can learn broad ordering from
the independently qualified historical action teacher, but winner-only labels
did not concentrate the selected action reliably enough for promotion.
ADRs 0050-0052 then rejected cross-attention, a trained immediate-rank
residual, and exact hex-rotation augmentation on the same immutable corpus.

That corpus stores only the selected action as positive. It discards the
teacher's rollout mean, rollout variance, sample allocation, and distinction
between evaluated and unevaluated actions. A replay audit also found that its
64 retained actions contained only 595 of 2,344 R2 teacher estimates in one
complete game, or 25.38%. Enriching the old corpus would therefore preserve
too little of the teacher's decision evidence.

During the audit, cross-process teacher replay exposed randomized
`HashMap::values()` order in the historical NNUE prefilter. Market
representatives and all score ties now have total deterministic ordering. An
in-process parity command compares the original selection path with the
instrumented estimate path from the same public state and rejects any action
drift.

## Decision

Collect a fresh paired corpus in one teacher pass:

1. Run the exact qualified K32 historical heuristic with R600, LMR enabled,
   diverse prefiltering enabled, and no paid prelude.
2. At each canonical V2 decision, retain the selected action, every teacher
   frontier action, the complete V2 pattern frontier, immediate top 16, and
   BLAKE3-ordered legal negatives up to 96 actions.
3. Write the shared state and explicit action rows to one-game `.cim` source
   shards.
4. Write a hash-aligned `.imv` sidecar with per-action rollout mean, rollout
   standard deviation, sample count, source flags, and selected bit.
5. Require every teacher estimate to align with one retained canonical action.
   Source and target manifests, group IDs, action indices, candidate counts,
   action hashes, selected bits, shard ranges, and checksums must agree.
6. Resume only at one-game boundaries. If a crash leaves the source one shard
   ahead, replay that game and require byte-exact source equality before
   appending its target shard.

The teacher identity is
`canonical-action-legacy-heuristic-deterministic-v2-k32-r600-lmr-no-paid-prelude`.
The historical evaluator remains non-promotable; only canonical action
evidence is used.

## Frozen MLX Experiment

- Train domain: split `train`, indices 51,000-51,063, 64 games.
- Validation domain: split `validation`, indices 51,000-51,015, 16 games.
- Test and gameplay domains: unopened and unauthorized.
- Candidate contract: 96 actions, immediate top 16, complete pattern and K32
  teacher frontiers, deterministic BLAKE3 negatives.
- Teacher: deterministic K32/R600/LMR policy and
  `nnue_weights_v4opp_modal_iter3.bin`.
- Model: fresh `shared-state-action-imitation-v1`, hidden 96, four attention
  heads, two board blocks, one market block.
- Warm start: prohibited.
- Pair target: sigmoid of the rollout-mean difference divided by combined
  standard error with a fixed 1.0-point variance floor.
- Pair weighting: `0.25 + 0.75 * abs(2p - 1)`.
- Auxiliary target: selected-action listwise cross-entropy with coefficient
  0.25 across all retained actions.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Training: batch 16, seed 20260616, at most 20 epochs, validation patience
  five, validation distributional loss selects the checkpoint.
- Commands: `make collect-imitation-evidence`,
  `make train-imitation-distribution`, and
  `make resume-imitation-distribution`.

The selected checkpoint advances only if all of these validation gates pass:

- exact dataset, source, target, checkpoint, and resume integrity;
- 100% alignment of teacher estimates in both splits;
- distributional loss below the untouched initialization;
- selected-action top-one accuracy at least 23%;
- selected-action top-five recall at least 58%;
- selected-action MRR at least 0.40;
- predicted top action belongs to the scored teacher frontier in at least 90%
  of groups;
- scored-action pairwise accuracy at least 75%;
- scored value-difference correlation at least 0.35.

Conditional mean regret, scored top-one recall, pairwise log loss, pairwise
Brier score, and scored rank correlation are reported diagnostics and may not
replace a failed gate.

Passing authorizes only a separately preregistered fresh test collection and
evaluation. The ADR 0049 test split remains sealed and may not select or
qualify this successor. Missing any gate rejects this exact experiment before
test access, promotion, or gameplay.

## Maximum Compute

One local 64-game train collection, one local 16-game validation collection,
and one Apple-GPU training run of at most 20 epochs. Collection is resumable
but may not change source, executable, weights, teacher, candidate contract,
seed range, or rollout budget. No external compute, second seed sweep,
hyperparameter search, threshold change, warm start, test collection, or
gameplay benchmark is authorized by this ADR.

## Implementation Evidence Before Registration

- R2 instrumented parity: one complete train game, 80 decisions, 2,344
  candidate estimates, exact original-action agreement, 75.827 seconds.
- R2 train smoke: 80 groups, 7,680 actions, 2,344 of 2,344 estimates aligned,
  38.2 seconds.
- R2 validation smoke: 80 groups, 7,680 actions, 2,434 of 2,434 estimates
  aligned, 41.6 seconds.
- The paired collector completed a no-op resume after validation.
- MLX loaded both real paired datasets, completed a GPU epoch, restored the
  exact optimizer/cursor checkpoint, and improved smoke validation loss from
  1.835712 to 1.741989 after two epochs. These one-game results verify wiring
  only and are not strength evidence.

## Result

The mandatory prerequisite passed before substantive collection:

- one complete train game at index 90,000;
- R600 on both the original and instrumented paths;
- 80 decisions and 2,400 candidate estimates;
- 3 minimum and 228 maximum samples per candidate;
- exact selected-action agreement at every decision;
- 292.017 seconds elapsed;
- report:
  `docs/v2/reports/canonical-action-teacher-estimate-parity-r600.json`;
- report BLAKE3:
  `a8d7e6fd97a1a38a27e60119ac5b337be86b5dfcd5f10a10af7c7451bf385353`;
- executable BLAKE3:
  `c4c60afa7962af4582557128bfebeb30badde8771e6a83ed33eaae9e8c02e5b1`;
- weights BLAKE3:
  `9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400`.

The frozen corpus completed locally:

- train: 64 games, 5,120 groups, 491,520 actions, and
  153,021/153,021 estimates aligned in 9,426.0 seconds;
- validation: 16 games, 1,280 groups, 122,880 actions, and
  38,641/38,641 estimates aligned in 2,346.9 seconds.

The Apple-GPU run selected epoch six and stopped after five non-improving
epochs. Distributional loss improved from 1.832475 to 1.534834, and scored
value-difference correlation passed at 0.444333. Every remaining behavioral
gate failed:

- top one 13.750% versus 23%;
- top five 38.438% versus 58%;
- MRR 0.269223 versus 0.40;
- predicted teacher coverage 71.406% versus 90%;
- scored pairwise accuracy 67.975% versus 75%.

Train top one was only 17.461%, so this is not primarily validation overfit.
The model learns some rollout-value geometry but forgets useful exact
immediate ordering and chooses an unscored action in 28.594% of validation
groups. The report is
`docs/v2/reports/canonical-action-mce-distribution-v1-validation.md`.

ADR 0053 is rejected without test access.
