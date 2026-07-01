# Cascadia v3 100+ Plan Review Approval Record

Status: approved by Plan Review.
Date: 2026-07-01.
Workflow: `plan-build-verify`.
Reviewed plan: `/Users/johnherrick/cascadia/cascadiav3/V3_100_EXECUTION_PLAN.md`.
Progress ledger: `/Users/johnherrick/cascadia/cascadiav3/V3_100_PROGRESS_2026-07-01.md`.

This file is the durable record of what Plan Review approved. It is intentionally
not a training result, not a strength claim, and not a replacement for the
execution plan. It exists so implementation can continue from an explicit
approved contract rather than chat history.

## Approved Objective

Build a transformer-based Cascadia v3 policy for four-player
Card-A-equivalent games with no habitat bonuses that can eventually demonstrate:

- all-v3 mean score at least 100;
- game-block 95% lower confidence bound at least 100;
- promotion by paired gameplay evidence, not validation loss alone.

The approved route is literature-aligned and gameplay-gated:

1. Search-improved action-value learning.
2. Expert iteration.
3. Search-integrated serving.
4. Promotion only by paired gameplay evidence.

Validation loss, imitation accuracy, GPU utilization, and successful completion
of a training job are diagnostics only. They do not approve a checkpoint for
promotion.

## Approved Scope

The reviewed implementation scope is primarily:

- `/Users/johnherrick/cascadia/cascadiav3/`
- `/Users/johnherrick/cascadia/docs/v3/`

Large-scale training data must use packed tensor `.npz` paths. JSONL remains
acceptable only for tiny audit fixtures and debugging. Production expert
iteration should not route through large JSONL shards.

The approved board representation is the existing radius-6 CascadiaFormer path.
Radius 6 is large enough for the observed game envelope and avoids spending
capacity on unused hex space. Merging this effort with any separate larger-board
campaign requires separate approval.

## Approved First Slice: EI-0 Search Bootstrap

Plan Review approved implementing the EI-0 search-bootstrap pipeline before
claiming any transformer playing-strength result.

Required Rust exporter work:

- add mode `--greedy-state-search-bootstrap-tensor-corpus`;
- generate roots from greedy self-play states;
- evaluate greedy-ranked K32/K64 candidate menus with rollout labels;
- keep the greedy action at action index 0;
- set `selected_action` to the rollout-best action;
- advance the actual game trajectory with the greedy action;
- write packed `cascadiav3.expert_tensor_shard.v1` `.npz` shards;
- emit Q, score-to-go, variance/count, visits, exact afterstate scores, final
  score decomposition, and teacher-vs-greedy advantage metadata.

The approved reason for this slice is narrow: establish whether a transformer
can learn search-improved action values from greedy-state roots without
immediately paying the full cost of model-state expert iteration.

## Approved Q Semantics

Plan Review approved defining the transformer Q head as predicted active-seat
score-to-go, not raw final score.

Every Q-serving path must rank by:

```text
derived_final_q = exact_afterstate_score_active + predicted_score_to_go
```

Approved serving contract:

- interactive/search roots expose `exact_afterstate_score_active` aligned
  one-to-one with `legal_actions`;
- inference request collation requires and preserves exact afterstate scores;
- model evaluation returns raw `score_to_go` and derived final `q`;
- benchmark and search-serving Q paths rank with derived final Q;
- a synthetic rank-flip test proves raw score-to-go ranking would select the
  wrong action.

## Approved Training Objective

Plan Review approved adding trainer objective:

```bash
--objective search-improved-greedy-retention
```

Initial approved loss weights:

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

Approved objective semantics:

- train `q_head` against `target_score_to_go`;
- compute Q ranking and regret using derived final Q;
- confidence-weight Q examples using rollout variance/count;
- clamp confidence weights to a stable range;
- keep greedy retention as a guardrail, not the main objective;
- log score-to-go Q loss, final-Q regret, teacher advantage over greedy,
  weighted policy loss, greedy top-1, teacher rank, and greedy rank.

## Approved Filter And Data Safety

Training is not allowed to silently drop the teacher-selected action.

Approved filter rules:

- raw K32 to filtered K32 may use `greedy-prefix-strict`;
- any raw K greater than filtered K must use a selected-preserving filter such
  as `greedy-prefix-with-selected`;
- filter summaries must report `selected_action_dropped_count == 0`;
- the runner must fail before training if selected labels are dropped.

Every valid action in an accepted shard must satisfy:

```text
target_q == exact_afterstate_score_active + target_score_to_go
```

## Approved Warm-Start And Resume Contract

Plan Review approved adding:

```bash
--init-manifest
--resume
```

Approved semantics:

- `--init-manifest` loads model weights only;
- `--resume` restores optimizer, scheduler, scaler, RNG state, and loader
  cursor;
- exact resume must refuse mismatches in dataset manifests, schema ids, model
  config, batch size, gradient accumulation, optimizer, seed, source hashes,
  objective, and loss weights.

## Approved Performance And Resource Gates

Runbooks must contain positive throughput fields:

- `roots_per_second`
- `rollout_evals_per_second`
- `bytes_per_record`
- `train_step_seconds`

Search/game benchmark reports must support and enforce:

```bash
--max-treatment-control-time-ratio 1.20
```

A checkpoint cannot be promoted if treatment/control aggregate decision-time
ratio exceeds `1.20`, even if its score looks promising.

## Approved Search-Integrated Serving Gate

Plan Review approved adding a CascadiaFormer search benchmark that uses model
policy and derived final Q as K24/K32/K64 prefilters into sampled search.

Reports must include:

- retained count;
- full-search winner retention;
- search regret;
- paired score delta;
- decision seconds;
- score breakdown;
- treatment/control timing ratio.

Direct no-search policy strength is useful, but it is not sufficient for final
promotion unless search-integrated play also passes.

## Approved EI-0 Run

After smoke validation passes, the reviewed EI-0 run is:

- train roots: 20,000;
- locked validation roots: 4,000;
- candidate menu: K32;
- rollouts/action: 4;
- rollout top-k: 4;
- tensor format: stored `.npz`;
- relation cache: materialized relation-tail;
- model: CascadiaFormer-S;
- batch size: 192 or 256;
- gradient accumulation: 1;
- steps: 25,000;
- optimizer: AdamW `(0.9, 0.95)`;
- weight decay: `0.05`;
- warmup: 2%;
- schedule: cosine decay to 10% of initial LR;
- SWA: final 20%;
- LR: `1e-4` when warm-starting from
  `cascadiav3/checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json`;
- LR: `2e-4` only for a from-scratch fallback.

## Approved EI-1, EI-2, And Scale-Up Conditions

Plan Review approved advancing beyond EI-0 only after gameplay/search gates show
real benefit.

For EI-1/EI-2:

- collect 50,000 train roots and 10,000 validation roots from the promoted
  checkpoint plus greedy/prior pool;
- use K64 union menus from greedy top32 plus model top32;
- start with R8/R16 labels;
- use R32 on high-disagreement roots;
- train CascadiaFormer-S for 50,000 steps;
- mix data as 50% current EI, 30% prior EI, 20% bootstrap.

Scaling to CascadiaFormer-M is approved only after CascadiaFormer-S gives at
least `+0.25` paired search-integrated gain over 500 games or clears the 97
gate.

Candidate CascadiaFormer-M shape:

- 12 layers;
- `d_model=768`;
- 12 heads;
- FFN 3072;
- gradient checkpointing;
- effective batch 256;
- LR `5e-5`;
- 80,000 steps.

## Approved Gameplay Gates

Staged gates:

- 95 gate: search-integrated v3 reaches at least 95 mean over at least 100
  matched games with no wildlife/category allocation collapse.
- 97 gate: promoted checkpoint beats incumbent/control by at least `+0.25`
  paired mean over 250-500 pairs and reaches at least 97 mean.
- 100 gate: freeze champion, run 1,000 all-v3 games, then extend to 4,000 if
  the confidence interval can plausibly cross 100.

EI-0 promotion requires at least one of:

- 100-game paired delta versus greedy is positive;
- 500-game paired delta versus greedy is at least `+0.25`;
- search-integrated play improves over the full-search/greedy-search baseline.

Reject or hold if:

- rollout teacher advantage is near zero;
- direct policy is worse than greedy by more than 1.5 points;
- validation Q/policy do not improve;
- greedy retention collapses below 0.20 after 2,000 steps;
- animal allocation collapses;
- treatment/control aggregate decision-time ratio exceeds 1.20.

Allocation collapse means a run improves one category, especially Bear, while
reducing total non-Bear wildlife and total score.

## Approved Validation Commands

No browser checks are required for this plan.

Core validation:

```bash
git status --short -- docs/v3 cascadiav3
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_schema_registry --include-legacy --include-expert
```

Exporter and tensor invariant validation:

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
  --val cascadiav3/fixtures/ei0_tiny_tensor.npz \
  --train-format npz --val-format npz \
  --steps 4 --batch-size 2 --device cpu \
  --objective search-improved-greedy-retention \
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

## Explicit Non-Approvals

Plan Review did not approve claiming success from:

- validation loss alone;
- imitation accuracy alone;
- no-search policy strength alone;
- engineering smoke tests;
- GPU utilization or training completion alone;
- a run that improves one wildlife category while collapsing total score or
  total non-Bear wildlife;
- scaling to CascadiaFormer-M before CascadiaFormer-S earns a search-integrated
  gameplay gain;
- reviving large JSONL generation as the main expert-iteration path.

## Residual Risks Accepted By Review

- EI-0 K32/R4 may produce weak teacher advantage; scaling is blocked if that
  happens.
- Q-serving touches multiple request and serving paths, so invariant tests are
  mandatory.
- Search-integrated gains remain empirical and must be proven by paired
  gameplay gates.
- Strong no-search Q performance may still fail to retain the full-search
  winner, so search-integrated evaluation is required before promotion.

## Current Operating Instruction

Proceed with implementation, smoke validation, EI-0 training, gameplay
benchmarking, and only then scale to model-state expert iteration or larger
models under the gates above.

Track live status in:

```text
/Users/johnherrick/cascadia/cascadiav3/V3_100_PROGRESS_2026-07-01.md
```
