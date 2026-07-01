# CascadiaFormer Training Pipeline

This is the canonical v3 runbook. It replaces the older scattered planning
files and keeps the training method in one place.

## Data Formats

Use packed tensor shards for real training:

- `greedy_policy_tensor_shard_v1`: compact greedy behavior-cloning shards.
- `cascadiav3.expert_tensor_shard.v1`: packed expert roots from Rust.
- `relation_tail` shards: filtered fixed-capacity action relation caches used
  by the GPU trainer.

Use JSONL only for tiny audit fixtures that need human-readable reconstruction
and public-boundary validation.

Packed expert shards contain:

- public token features;
- semantic action features;
- per-root action offsets;
- selected action labels;
- per-action Q targets, Q masks, variance/counts, visits, priors;
- exact afterstate score for the active seat;
- score decomposition labels;
- final score/rank vectors;
- sparse relation edges plus materialized relation-tail tensors.

## Target Semantics

For every legal action:

```text
per_action_Q = estimated active-seat final raw score
target_score_to_go = per_action_Q - exact_afterstate_score_active
```

The model's `q_head` predicts score-to-go. Serving derives final Q by adding the
exact afterstate score back in. Losses must ignore invalid Q targets. Reports
must separate raw score-to-go error from derived-final-Q regret.

## Current Baseline

The transformer has reached the greedy neighborhood but has not beaten greedy:

- corrected greedy-state K32 retention run:
  - locked validation greedy top-1: `0.6780`;
  - 100-game complete-game benchmark: model `86.7800`, greedy `87.5875`;
  - paired delta: `-0.8075`;
  - exact greedy-action match: `67.3625%`.

This is enough to validate the tensor/model plumbing. It is not strength
evidence.

## Stage 0: Contracts

Required CPU gates:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_schema_registry --include-legacy --include-expert
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
```

Core invariants:

- legal-action coverage is exact;
- selected actions are never dropped by filters;
- public observations do not leak hidden future stack order;
- D6 transforms round-trip every spatial token and action pointer;
- category targets sum to final score;
- JSONL audit fixtures reconstruct from seed plus replay prefix;
- packed tensor shards pass invariant validators before training.

## Stage 1: Greedy Copy Baseline

Purpose: prove the model can stay in-distribution and reproduce a competent
policy surface before asking it to improve.

Recommended shape:

- generate actual greedy self-play roots;
- use greedy-ranked K32 menus;
- keep greedy action at index 0;
- optimize greedy policy, greedy margin, and score-to-go diagnostics;
- benchmark complete games against greedy, not only offline root accuracy.

Promotion out of this stage requires nonregression against greedy in gameplay.

## Stage 2: EI-0 Search Bootstrap

EI-0 is the first useful attempt to improve beyond greedy.

Exporter mode:

```bash
--greedy-state-search-bootstrap-tensor-corpus
```

Semantics:

- roots come from greedy self-play states;
- candidate menus are greedy-ranked K32 or K64;
- the greedy action remains index 0;
- search labels select the rollout-best action;
- the actual trajectory advances with the greedy action;
- tensors include Q, score-to-go, variance/counts, visits, exact afterstate
  score, final decomposition, and teacher-vs-greedy advantage metadata.

Initial training run:

- train roots: `20,000`;
- locked validation roots: `4,000`;
- candidate menu: `K32`;
- rollouts/action: `4`;
- rollout top-k: `4`;
- tensor format: stored `.npz`;
- filter: `greedy-prefix-strict`;
- model: CascadiaFormer-S;
- batch size: `192` or `256`;
- gradient accumulation: `1`;
- optimizer: AdamW `(0.9, 0.95)`;
- weight decay: `0.05`;
- warmup: `2%`;
- schedule: cosine decay to `10%` of initial LR;
- SWA: final `20%`;
- LR: `1e-4` from greedy-retention warm start, `2e-4` from scratch fallback;
- steps: `25,000`.

Objective preset: `search-improved-greedy-retention`.

Initial loss weights:

```text
policy: 1.00
q: 0.20
value: 0.05
score: 0.02
rank: 0.01
uncertainty: 0.01
greedy_policy: 0.75
greedy_margin: 0.25
greedy_margin_value: 0.25
```

Teacher confidence should use Q variance/counts, standard-error weighting, and
clamping in `[0.25, 4.0]`.

## Stage 3: Expert Iteration

Expert iteration starts only after EI-0 passes gameplay smoke and
search-integrated gates.

Cycle shape:

- 10 cycles;
- 10k games/cycle;
- newest model occupies one rotated focal seat;
- opponents are frozen control/champion plus prior v3 pool;
- cycle 1 uses all frozen control opponents;
- label 10k roots/cycle for cycles 1-3;
- label 20k roots/cycle for cycles 4-10;
- exploration by cycle:
  `0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.02`;
- train two origins per cycle;
- pass LRs: `3e-5`, `3e-5`, `1e-5`;
- data mix: 50% current cycle, 30% prior three cycles, 20% bootstrap/older.

Rejected or inconclusive candidates do not halt the campaign. The incumbent
remains champion and the rejected model may still be retained as historical
opponent diversity.

## Checkpoint Contract

Save every 1k optimizer steps and at every epoch/block boundary:

- model weights;
- optimizer, scheduler, scaler;
- RNG states;
- resume-safe loader cursor;
- schema ids;
- dataset manifests and checksums;
- source hashes;
- search config;
- objective and loss weights;
- metrics history;
- model manifest.

Also save:

- best locked-validation checkpoint;
- SWA checkpoint over the final 20%;
- small JSON summary for every run;
- metrics JSONL for loss curves and later plotting.

`--init-manifest` loads weights only. `--resume` restores the complete training
state and refuses mismatches in source, dataset, schema, model config, batch,
grad accumulation, objective, optimizer, seed, or loss weights.

## Gameplay Gates

Do not promote on loss alone.

Gates:

- 95 gate: search-integrated v3 reaches at least 95 mean over at least 100
  matched games without wildlife/category allocation collapse.
- 97 gate: promoted checkpoint beats incumbent/control by at least +0.25 paired
  mean over 250-500 pairs and reaches at least 97 mean.
- 100 gate: freeze champion, run 1,000 all-v3 games, then extend to 4,000 if
  the confidence interval can plausibly cross 100.

Search reports must include:

- retained count;
- full-search winner retention;
- search regret;
- paired delta;
- decision seconds;
- score breakdown;
- treatment/control timing ratio.

Treatment/control aggregate decision-time ratio above `1.20` is a resource
regression unless explicitly approved before the run.
