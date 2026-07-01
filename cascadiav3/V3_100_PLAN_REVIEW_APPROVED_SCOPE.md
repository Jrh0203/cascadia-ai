# Cascadia v3 Plan Review Approved Scope

Date: 2026-07-01.
Status: Plan Review approved.
Workflow: `plan-build-verify`.

Source of truth:

- Full execution plan:
  `/Users/johnherrick/cascadia/cascadiav3/V3_100_EXECUTION_PLAN.md`
- Full approval record:
  `/Users/johnherrick/cascadia/cascadiav3/V3_100_PLAN_REVIEW_APPROVAL.md`
- Live progress ledger:
  `/Users/johnherrick/cascadia/cascadiav3/V3_100_PROGRESS_2026-07-01.md`

This file records the concise contract approved by Plan Review. It is not a
training result, not a checkpoint promotion, and not evidence that Cascadia v3
already scores 100. It exists so future implementation and experiment decisions
can be checked against the approved scope without relying on chat history.

## Approved Objective

Build a transformer-based Cascadia v3 policy for four-player
Card-A-equivalent games with no habitat bonuses that can eventually demonstrate:

- all-v3 mean score at least 100;
- game-block 95% lower confidence bound at least 100;
- promotion by paired gameplay evidence rather than validation loss alone.

The approved route is:

1. Search-improved action-value learning.
2. Expert iteration.
3. Search-integrated serving.
4. Promotion only by paired gameplay evidence.

Validation loss, imitation accuracy, no-search strength, training completion,
and GPU utilization are diagnostics only.

## Approved Scope Boundaries

Implementation scope is primarily:

- `/Users/johnherrick/cascadia/cascadiav3/`
- `/Users/johnherrick/cascadia/docs/v3/`

Approved storage format:

- packed expert tensor `.npz` for real training data;
- JSONL only for tiny fixtures, audits, and debugging.

Approved board envelope:

- radius-6 CascadiaFormer path;
- no excessive hex space unless a later reviewed plan expands the
  representation.

Any merge with a separate larger-board or NNUE campaign requires separate
approval.

## Approved First Slice

The approved first implementation slice was EI-0 greedy-state search bootstrap.

Required exporter behavior:

- add a Rust exporter mode equivalent to
  `--greedy-state-search-bootstrap-tensor-corpus`;
- generate roots from greedy self-play states;
- evaluate greedy-ranked K32/K64 candidate menus with rollout labels;
- keep the greedy action at action index 0;
- set `selected_action` to the rollout-best action;
- advance the actual game trajectory with the greedy action;
- write packed `cascadiav3.expert_tensor_shard.v1` `.npz` shards;
- emit Q, score-to-go, variance/count, visits, exact afterstate scores, final
  score decomposition, and teacher-vs-greedy advantage metadata.

The purpose of EI-0 is to test whether a transformer can learn
search-improved action values from greedy-state roots before paying the full
cost of model-state expert iteration.

## Approved Q Contract

The transformer Q head is approved as predicted active-seat score-to-go.

Every Q-serving path must rank actions by:

```text
derived_final_q = exact_afterstate_score_active + predicted_score_to_go
```

Approved serving requirements:

- roots expose `exact_afterstate_score_active` aligned one-to-one with
  `legal_actions`;
- inference request collation requires and preserves exact afterstate scores;
- model evaluation returns both raw `score_to_go` and derived final `q`;
- benchmark/search serving ranks by derived final Q;
- a synthetic rank-flip test fails if raw score-to-go is used for ranking.

## Approved Training Objective

Approved trainer objective:

```bash
--objective search-improved-greedy-retention
```

Approved initial loss weights:

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

Approved semantics:

- train `q_head` against `target_score_to_go`;
- compute Q ranking and regret using derived final Q;
- confidence-weight Q examples using rollout variance/count;
- keep greedy retention as a guardrail, not the primary objective;
- log score-to-go Q loss, final-Q regret, teacher advantage over greedy,
  weighted policy loss, greedy top-1, teacher rank, and greedy rank.

## Approved Safety Gates

Training data safety:

- `selected_action_dropped_count` must be reported;
- accepted training shards must have `selected_action_dropped_count == 0`;
- raw K32 to filtered K32 may use `greedy-prefix-strict`;
- any raw K greater than filtered K must preserve the selected action;
- every valid action in an accepted shard must satisfy:

```text
target_q == exact_afterstate_score_active + target_score_to_go
```

Resume safety:

- `--init-manifest` loads model weights only;
- `--resume` restores optimizer, scheduler, scaler, RNG state, and loader
  cursor;
- exact resume refuses mismatches in dataset manifests, schema ids, model
  config, batch size, gradient accumulation, optimizer, seed, source hashes,
  objective, or loss weights.

Performance safety:

- runbooks must include positive `roots_per_second`,
  `rollout_evals_per_second`, `bytes_per_record`, and `train_step_seconds`;
- search/game benchmark promotion is blocked if treatment/control aggregate
  decision-time ratio exceeds `1.20`.

## Approved EI-0 Run Shape

After smoke validation passes, the approved EI-0 run shape is:

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
- LR: `1e-4` when warm-starting from an approved greedy-retention checkpoint;
- LR: `2e-4` only for a from-scratch fallback.

## Approved EI-1/EI-2 And Scale Conditions

Advancing beyond EI-0 is approved only after gameplay/search gates show real
benefit.

Approved EI-1/EI-2 shape:

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

## Approved Gameplay Gates

The staged gates are:

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
- treatment/control aggregate decision-time ratio exceeds `1.20`.

Allocation collapse means a run improves one category, especially Bear, while
reducing total non-Bear wildlife and total score.

## Explicit Non-Approvals

Plan Review did not approve:

- claiming success from validation loss alone;
- claiming success from imitation accuracy alone;
- claiming success from no-search policy strength alone;
- claiming success from engineering smoke tests;
- claiming success from GPU utilization or training completion;
- promoting a run that improves one category while reducing total score or
  total non-Bear wildlife;
- scaling to CascadiaFormer-M before CascadiaFormer-S earns a
  search-integrated gameplay gain;
- reviving large JSONL generation as the main expert-iteration path.

## Operating Instruction

Proceed with implementation, smoke validation, EI-0/EI-N training, gameplay
benchmarking, and scaling only under the gates above.

Track live evidence and deviations in:

```text
/Users/johnherrick/cascadia/cascadiav3/V3_100_PROGRESS_2026-07-01.md
```
