# Cascadia v3 100+ Execution Plan

Status: Plan Review PASS.
Date: 2026-07-01.
Scope: `cascadiav3/` and `docs/v3/` unless a later approved implementation plan explicitly expands scope.

This is the reviewed plan for getting Cascadia v3 to consistent 100+ scoring.
It was approved by Plan Review in the `plan-build-verify` workflow. It is not
itself evidence that the system scores 100 yet, and it is not a substitute for
the required gameplay gates.

## Goal

Build a Cascadia v3 four-player Card-A-equivalent policy, with no habitat
bonuses, whose all-v3 mean is at least 100 and whose game-block 95% lower
confidence bound is also at least 100.

The path is literature-first and gameplay-gated:

1. Search-improved action-value learning.
2. Expert iteration.
3. Search-integrated serving.
4. Promotion only by paired gameplay evidence.

Validation loss, imitation accuracy, and no-search policy strength are useful
diagnostics, not promotion criteria by themselves.

## Staged Gameplay Gates

- 95 gate: search-integrated v3 reaches at least 95 mean over at least 100
  matched games with no wildlife/category allocation collapse.
- 97 gate: promoted checkpoint beats incumbent/control by at least +0.25 paired
  mean over 250-500 pairs and reaches at least 97 mean.
- 100 gate: freeze champion, run 1,000 all-v3 games, then extend to 4,000 if
  the confidence interval can plausibly cross 100.

## Current-State Evidence

- The scale path is packed expert tensor `.npz`, not large JSONL. JSONL remains
  for tiny audit fixtures.
  Source: `/Users/johnherrick/cascadia/docs/v3/FULL_V3_TRAINING_PIPELINE.md`.
- Target semantics are already defined as active-seat final raw Q plus
  score-to-go:
  `per_action_Q = active-seat final raw score`
  `per_action_score_to_go = per_action_Q - exact_afterstate_score_active`.
- The current transformer baseline is below greedy: CascadiaFormer policy mean
  about `86.7800` versus greedy `87.5875`, paired delta about `-0.8075`.
- The existing CascadiaFormer already has action-query logits, `q_head`,
  uncertainty, value, rank, and score-decomposition heads.
- Current trainer objective choices do not yet include the reviewed
  `search-improved-greedy-retention` preset.
- Current no-search Q serving ranks by raw model Q. After this plan, raw model Q
  means predicted score-to-go, so serving must rank by derived final Q:
  `exact_afterstate_score_active + predicted_score_to_go`.
- Current inference/interactive roots do not yet require aligned
  `exact_afterstate_score_active`; that is a required fix before Q-based gates
  can be trusted.

## Constraints

- Preserve unrelated dirty worktree changes.
- Keep implementation primarily under `cascadiav3/` and `docs/v3/`.
- Do not open protected seed domains for smoke tests.
- Do not claim final 100 evidence from engineering smoke or validation loss.
- Keep large training artifacts packed as `.npz`; do not revive JSONL for large
  training paths.
- Use the existing radius-6 CascadiaFormer path unless John explicitly chooses
  to merge this with the separate radius-7 NNUE campaign.

## Ordered Implementation Plan

### 1. Add EI-0 Rust Exporter Mode

Add `Mode::GreedyStateSearchBootstrapTensorCorpus` and CLI flag:

```bash
--greedy-state-search-bootstrap-tensor-corpus
```

The exporter must:

- Generate roots from greedy self-play states.
- Evaluate greedy-ranked K32/K64 candidate menus with rollout labels.
- Keep the greedy action at index 0.
- Set `selected_action` to the rollout-best action.
- Advance the actual game trajectory with the greedy action, not the
  rollout-best action.
- Write packed `cascadiav3.expert_tensor_shard.v1` `.npz` shards.
- Include Q, score-to-go, variance/count, visits, exact afterstate scores,
  final scores/decomposition, and teacher-vs-greedy advantage metadata.

### 2. Fix Q Semantics End To End

Define `outputs["q"]` as predicted score-to-go.

Every Q-serving path must rank by:

```text
derived_final_q = exact_afterstate_score_active + predicted_score_to_go
```

Required code-level contract:

- `build_interactive_root` exposes `exact_afterstate_score_active`, aligned
  one-to-one with `legal_actions`.
- `inference_request_view` requires `exact_afterstate_score_active`.
- Inference collation includes `exact_afterstate_score_active`.
- `torch_inference_bridge._model_eval` returns both:
  - `score_to_go`: raw model output.
  - `q`: derived final Q.
- `torch_cascadiaformer_game_benchmark.rank_root_with_model` and new
  search-serving paths rank Q selection by derived final Q.
- Add a synthetic rank-flip test where raw score-to-go and derived final Q pick
  different actions. The test must fail if serving ranks by raw score-to-go.

### 3. Add `search-improved-greedy-retention` Objective

Add parser choice:

```bash
--objective search-improved-greedy-retention
```

Initial preset loss weights:

```text
policy: 1.0
q: 0.20
value: 0.05
score: 0.02
rank: 0.01
uncertainty: 0.01
greedy_policy: 0.75
greedy_margin: 0.25
greedy_margin_value: 0.25
```

Training semantics:

- Train `q_head` against `target_score_to_go`, not raw final Q.
- Compute ranking and regret using derived final Q.
- Add confidence weighting from `q_count` and `q_variance`, using teacher
  standard error and clamp range `[0.25, 4.0]`.
- Log:
  - `locked_val_score_to_go_q`
  - `locked_val_final_q_regret`
  - `locked_val_teacher_advantage_over_greedy`
  - `locked_val_weighted_policy`
  - `locked_val_greedy_top1`
  - mean teacher rank
  - mean greedy rank

### 4. Add Filter Safety

Training shards must report:

```text
selected_action_dropped_count == 0
```

Rules:

- EI-0 K32 raw to K32 filtered may use `greedy-prefix-strict`.
- Any raw K greater than filtered K must use `greedy-prefix-with-selected` or
  another selected-preserving mode.
- Runner fails before training if selected labels are dropped.

### 5. Add Warm-Start And Exact Resume

Add:

```bash
--init-manifest
--resume
```

Semantics:

- `--init-manifest` loads model weights only.
- `--resume` restores optimizer, scheduler, scaler, RNG state, and loader
  cursor.
- Resume refuses mismatches in dataset manifests, schema ids, model config,
  batch size, grad accumulation, optimizer, seed, source hashes, objective, or
  loss weights.

### 6. Add Performance Report Validation

Add a report checker, for example:

```bash
python3 -m cascadiav3.validate_runbook_performance
```

It must fail if required performance fields are missing or nonpositive.

Required runbook fields:

- `roots_per_second`
- `rollout_evals_per_second`
- `bytes_per_record`
- `train_step_seconds`

Search/game benchmark reports must also support:

```bash
--max-treatment-control-time-ratio 1.20
```

Promotion blocks if treatment/control aggregate decision-time ratio exceeds
`1.20`.

### 7. Add Search-Integrated Serving Gates

Extend or add a CascadiaFormer search benchmark that uses model policy and
derived final Q as K24/K32/K64 prefilters into sampled search.

Reports must include:

- retained count
- full-search winner retention
- search regret
- paired delta
- decision seconds
- score breakdown
- treatment/control timing ratio

Direct no-search policy strength is not enough to promote unless
search-integrated play also passes.

## EI-0 Training Run

Use this after smoke validation passes:

- Train roots: 20,000.
- Locked validation roots: 4,000.
- Candidate menu: K32.
- Rollouts/action: 4.
- Rollout top-k: 4.
- Tensor format: stored `.npz`.
- Relation cache: materialized relation-tail.
- Model: CascadiaFormer-S.
- Batch size: 192 or 256.
- Gradient accumulation: 1.
- Steps: 25,000.
- Optimizer: AdamW `(0.9, 0.95)`.
- Weight decay: `0.05`.
- Warmup: 2%.
- Schedule: cosine decay to 10% of initial LR.
- SWA: final 20%.
- LR: `1e-4` when warm-starting from
  `cascadiav3/checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json`.
- LR: `2e-4` only for from-scratch fallback.

Launch shape:

```bash
JOB_SLUG=cascadiaformer_ei0_greedy_search_bootstrap \
PROFILE=ei0_greedy_search_bootstrap \
EXPERT_TENSOR_MODE=greedy_search_bootstrap \
MAX_ACTIONS=32 FILTER_TOP_K=32 FILTER_MODE=greedy-prefix-strict \
OBJECTIVE=search-improved-greedy-retention MODEL_SIZE=S \
TRAIN_SEED_COUNT=250 VAL_SEED_COUNT=50 PLIES_PER_SEED=80 \
ROLLOUTS_PER_ACTION=4 ROLLOUT_TOP_K=4 TRAIN_STEPS=25000 \
BATCH_SIZE=192 GRAD_ACCUM=1 LR=0.0001 VAL_MAX_BATCHES=0 \
INIT_MANIFEST=cascadiav3/checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json \
bash cascadiav3/scripts/run_full_v3_training_pipeline.sh launch
```

## EI-1, EI-2, And Scale-Up

Promote to EI-1 only after gameplay/search gate success.

EI-1/EI-2:

- Collect 50,000 train roots and 10,000 validation roots from the promoted
  checkpoint plus greedy/prior pool.
- Use K64 union menus from greedy top32 plus model top32.
- Start with R8/R16 labels, then use R32 on high-disagreement roots.
- Train CascadiaFormer-S for 50,000 steps.
- Data mix: 50% current EI, 30% prior EI, 20% bootstrap.

Scale to CascadiaFormer-M only after CascadiaFormer-S gives at least +0.25
paired search-integrated gain over 500 games or clears the 97 gate.

Candidate M config:

- 12 layers.
- `d_model=768`.
- 12 heads.
- FFN 3072.
- Gradient checkpointing.
- Effective batch 256.
- LR `5e-5`.
- 80,000 steps.

## Acceptance Criteria For First Code Slice

The first implementation slice passes only when:

- New exporter mode produces nonempty `.npz` shards.
- `selected_action_index` is rollout-best.
- Greedy action remains index 0.
- Actual trajectory advances by greedy action.
- Every valid action satisfies:
  `target_q == exact_afterstate_score_active + target_score_to_go`.
- Interactive roots expose `exact_afterstate_score_active` aligned with
  `legal_actions`.
- Inference and benchmark Q paths rank by derived final Q.
- A rank-flip test proves raw score-to-go ranking would be wrong.
- `search-improved-greedy-retention` is accepted by the trainer CLI.
- The objective reports configured weights and has unit coverage.
- Filter summaries show `selected_action_dropped_count == 0`.
- Warm-start/resume positive and negative tests pass.
- Performance checker passes and blocks malformed or resource-regressed reports.

## Promotion Criteria

EI-0 promotes only if at least one is true:

- 100-game paired delta versus greedy is positive.
- 500-game paired delta versus greedy is at least +0.25.
- Search-integrated play improves over the full-search/greedy-search baseline.

Reject if:

- Rollout teacher advantage is near zero.
- Direct policy is worse than greedy by more than 1.5 points.
- Validation Q/policy do not improve.
- Greedy retention collapses below 0.20 after 2,000 steps.
- Allocation collapses.
- Treatment/control timing ratio exceeds 1.20.

Allocation collapse means a run improves one category, especially Bear, while
reducing total non-Bear wildlife and total score.

## Validation Commands

No browser checks are required.

```bash
git status --short -- docs/v3 cascadiav3
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_schema_registry --include-legacy --include-expert
```

Exporter and tensor invariant:

```bash
BINARY=cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter
$BINARY --greedy-state-search-bootstrap-tensor-corpus \
  --first-seed 2026410000 --seed-count 2 --plies-per-seed 2 \
  --max-actions 32 --rollouts-per-action 2 --rollout-top-k 4 \
  --tensor-compression stored \
  --out cascadiav3/fixtures/ei0_tiny_tensor.npz \
  --manifest cascadiav3/fixtures/ei0_tiny_tensor_manifest.json
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.expert_tensor_shards \
  --summarize-shard cascadiav3/fixtures/ei0_tiny_tensor.npz \
  --report cascadiav3/reports/ei0_tiny_tensor_summary.json
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_expert_tensor_invariants \
  --shard cascadiav3/fixtures/ei0_tiny_tensor.npz \
  --require-selected-action-dropped-count 0 \
  --require-q-equals-afterstate-plus-score-to-go
```

Q-serving invariant and objective smoke:

```bash
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_q_serving_semantics
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_cascadiaformer \
  --model-size tiny \
  --train cascadiav3/fixtures/ei0_tiny_tensor.npz \
  --val cascadiav3/fixtures/ei0_tiny_tensor.npz \
  --train-format npz --val-format npz \
  --steps 20 --batch-size 2 --device cpu \
  --objective search-improved-greedy-retention \
  --val-max-batches 1 \
  --out cascadiav3/reports/ei0_tiny_train_smoke.json
```

Warm-start/resume validation:

```bash
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_cascadiaformer \
  --model-size tiny --train cascadiav3/fixtures/ei0_tiny_tensor.npz \
  --val cascadiav3/fixtures/ei0_tiny_tensor.npz --train-format npz --val-format npz \
  --steps 4 --batch-size 2 --device cpu --objective search-improved-greedy-retention \
  --checkpoint-dir cascadiav3/checkpoints/resume_tiny_base \
  --out cascadiav3/reports/resume_tiny_base.json

PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_training_resume \
  --dataset cascadiav3/fixtures/ei0_tiny_tensor.npz \
  --init-manifest cascadiav3/checkpoints/resume_tiny_base/best_locked_val.manifest.json \
  --steps-before-resume 4 --steps-after-resume 8 \
  --batch-size 2 --seed 20260630 \
  --expect-exact-final-weight-match \
  --expect-mismatch-refusal dataset,model_size,batch_size,seed,objective,source_hash
```

Performance validation:

```bash
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_runbook_performance \
  --runbook cascadiav3/reports/full_v3_ei0_greedy_search_bootstrap_runbook.json \
  --require-positive roots_per_second,rollout_evals_per_second,bytes_per_record,train_step_seconds
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_runbook_performance \
  --benchmark cascadiav3/reports/cascadiaformer_ei0_search_game_benchmark.json \
  --max-treatment-control-time-ratio 1.20
```

## Risks And Mitigations

- Distribution shift: EI-0 labels greedy-state roots while advancing actual
  games by greedy action.
- Q semantic mismatch: blocked by interactive root contract, inference
  contract, derived-Q serving, and rank-flip tests.
- Greedy clone ceiling: greedy retention is a guardrail, not the objective.
- Weak rollout teacher: if K32/R4 has little advantage, increase teacher quality
  before scaling model.
- Filter label loss: training is blocked unless selected labels survive
  filtering.
- Raw-score instability: train score-to-go and rank by derived final Q.
- JSONL bloat: large paths stay `.npz`; bytes/root is validated.
- Low GPU utilization: relation-tail shards and stored `.npz` remain required;
  performance report must show nonzero throughput fields.
- Search latency regression: treatment/control ratio greater than 1.20 blocks
  promotion.
- Dirty worktree: implementation must preserve unrelated changes.

## Plan Review Result

Plan Review verdict: PASS.

Residual risks from Plan Review:

- EI-0 K32/R4 may still produce weak teacher advantage; this plan blocks scaling
  if that happens.
- The Q-serving change touches multiple request and serving paths, so the
  synthetic rank-flip test and invariant validator are required.
- Search-integrated gains remain empirical; this plan does not overclaim final
  100 evidence from smoke runs.
