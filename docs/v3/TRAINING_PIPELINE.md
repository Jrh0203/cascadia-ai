# CascadiaFormer Training Pipeline

This is the canonical v3 runbook. It replaces the older scattered planning
files and keeps the training method in one place.

## Ruleset Gate

Every corpus, checkpoint, report, and promotion comparison must identify the
rules semantics used to generate it. The active contract is
[`RULES_CONTRACT.md`](RULES_CONTRACT.md), semantics ID
`cascadia-base-official-2026-07-09`.

The optional free three-of-a-kind refresh became a real policy decision on
2026-07-08; the 2026-07-09 correction additionally enforces decision → hidden
chance draw → draft, so a policy cannot condition accept/decline on the actual
replacement. Earlier forced-refresh artifacts are legacy evidence and cannot
be mixed with corrected games in promotion statistics. Rebaseline every
opponent and incumbent after a rules-semantics change.

## Data Formats

Use packed tensor shards for real training:

- `greedy_policy_tensor_shard_v1`: compact greedy behavior-cloning shards.
- `cascadiav3.expert_tensor_shard.v1`: packed expert roots from Rust.
- `cascadiav3.expert_tensor_shard.v2`: v1 plus Gumbel self-play targets —
  action-aligned `improved_policy` soft targets and per-record
  `search_root_value`, with `final_score_vector`/`rank_vector`/decomposition
  labeled from real terminal outcomes instead of rollout means. Existing v2
  shards remain readable, but their exact-endgame rows and generation
  provenance cannot be reconstructed reliably after packing.
- `cascadiav3.expert_tensor_shard.v3`: adds an explicit per-record
  `exact_endgame` tensor and
  fail-closed metadata for ruleset, source revision, complete search and
  execution settings, exporter SHA/size, and teacher manifest/weights
  SHA/size. A v3 shard with fallback/unverified teacher identity is marked
  audit-only and the training corpus loader rejects it.
- `cascadiav3.expert_tensor_shard.v4`: the required format for new Gumbel
  generation. It adds `active_seat` and exact action-aligned
  wildlife/habitat/Nature afterstate components, with scalar-sum invariants,
  and is the only format accepted by structured-Q training.
- `relation_tail` shards: filtered fixed-capacity action relation caches used
  by the GPU trainer (v2-v4 fields pass through filtering and materialization;
  retained improved-policy slices are renormalized).

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
- explicit exact-endgame provenance for every v3+ root;
- for v4, the active seat plus exact per-action wildlife/habitat/Nature
  afterstate components;
- sparse relation edges plus materialized relation-tail tensors.

## Target Semantics

For every legal action:

```text
per_action_Q = estimated active-seat final raw score
target_score_to_go = per_action_Q - exact_afterstate_score_active
```

The legacy `q_head` predicts score-to-go. With `--q-decomposition`, the model
instead predicts three action-conditioned category residuals and defines
score-to-go as their sum. Serving derives final Q by adding the exact
afterstate score back in. Losses must ignore invalid Q targets. Reports must
separate raw score-to-go error from derived-final-Q regret.

Structured category supervision exists only for the selected real action:

```text
target_category_score_to_go =
    terminal_category_score(active_seat)
    - exact_selected_afterstate_category_score(active_seat)
```

Unselected actions retain ordinary completed-Q supervision on the summed
component output; assigning them the selected trajectory's terminal category
vector would be a false counterfactual label.

With `--q-quantiles K` (`K > 1`), the head is trained by pinball loss at
centered quantile levels `(k + 0.5) / K`; its ordinary `q` output remains the
arithmetic mean. Experimental q25/q50/q75 serving monotonically rearranges the
independent heads before interpolation, because pinball training does not
guarantee non-crossing outputs. Fixed-root probes diagnose crossing and action
rank changes, but only paired gameplay can establish a strength effect.

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

## Stage 3: Rollout-Teacher Expert Iteration (superseded)

The original Stage 3 (10 cycles of rollout-labeled roots) is superseded by
Stage 4. Its labels inherited the greedy-rollout teacher ceiling that the
K64/R32 test exposed; EI-1 was terminated mid-run on 2026-07-02. Rejected or
inconclusive candidates still do not halt any campaign: the incumbent remains
champion, and rejected models remain opponent-diversity material.

## Stage 4: Gumbel Self-Play Expert Iteration (active)

Canonical plan: [GUMBEL_SELFPLAY_CAMPAIGN.md](GUMBEL_SELFPLAY_CAMPAIGN.md).
Summary:

- Exporter mode: `--gumbel-selfplay-tensor-corpus` (new generation is schema
  v4; requires
  `--source-revision`). All four
  seats play via Gumbel top-m search with batched model leaf values over
  hidden-redeterminized states; every visited root exports completed-Q
  targets, `improved_policy`, `search_root_value`, and real-outcome value
  labels. Root menus are the full legal action set — never greedy-ranked.
- Trainer objective: `gumbel-selfplay` — soft-target policy loss against
  `improved_policy`, up-weighted value loss on real outcomes, no
  greedy-retention terms, `--max-example-passes 4` overfit guard.
- Cycle shape: ~1,250 train seeds x 80 plies (~100k roots) per cycle,
  125 validation seeds, warm start from the incumbent, LR `1e-4`, batch 192,
  SWA final 20%; replay window over the last 2-3 cycles via
  `TRAIN_SOURCE_WEIGHTS` (e.g. `1.0,0.5,0.25`).
- Leaf-value blend ramp `w`: 0.5 -> 0.75 -> 1.0 across cycles as the value
  head is retrained on real outcomes.
- Optional exact serving frontier: `--gumbel-exact-endgame-turns 1` replaces
  the final personal turn for each seat with complete-menu engine scoring.
  Exported rows identify `exact_endgame=true`, carry zero simulations and a
  one-hot improved policy, and therefore provide an exact zero-score-to-go
  terminal target rather than a model/search estimate.
- Before a shard becomes training input, independently verify its NPZ SHA and
  embedded v3+ provenance. Shards generated before the v3 contract are
  audit-only when the missing fields cannot be recovered from immutable
  launch evidence.
- Runner: `cascadiav3/scripts/run_gumbel_selfplay_cycle.sh` (delegates to
  the full pipeline with `EXPERT_TENSOR_MODE=gumbel_selfplay`).

### Structured-Q pilot

Schema v4 is the sole accepted input for the action-conditioned decomposition:

```text
--objective gumbel-selfplay-structured-q
--q-decomposition
--init-manifest <incumbent>
--init-skip-mismatched
--q-decomposition-head-only
```

The objective retains the normal Gumbel policy/value losses and completed-Q
loss, adding a 0.5-weight selected-action component loss. The first kill test
must be head-only: it freezes the trunk and every established head, trains only
`q_component_head`, and selects on untouched locked validation. Only after the
head clears its preregistered validation threshold should a full fine-tune or
paired gameplay battery consume GPU time. A lower component loss is not
promotion evidence.

The validation verdict is produced by
`python -m cascadiav3.torch_structured_q_probe`. Exact-endgame rows are
excluded from its primary read. All four gates must pass:

1. selected-action final-score RMSE improves at least 10% over the better of
   incumbent model Q and selected completed-Q teacher;
2. the 95% t-CI for paired candidate-minus-baseline absolute error is wholly
   below zero;
3. completed-Q RMSE over all retained q-valid actions is no more than 1.05x
   incumbent;
4. mean completed-Q regret increases by no more than 0.05 points.

Hyperparameters are selected on a disjoint v4 block. Run the probe once on a
third untouched seed block for the preregistered verdict; do not pick learning
rate or checkpoint from that final block.

### Pairwise comparator pilot

The confidence audit supports a bounded comparator experiment, not indiscriminate
all-pairs training. Use:

```text
--objective gumbel-selfplay-pairwise
--pairwise-comparator
--pairwise-rank 64
--pairwise-max-pairs-per-root 32
--pairwise-min-margin 0.25
--pairwise-min-snr 1.0
```

For the first kill test, warm-start the corrected incumbent with
`--init-skip-mismatched --pairwise-head-only`; this initializes only the new
head and freezes every established parameter. Select on locked-validation
pairwise loss/accuracy, then compare `--policy-mode logits` against
`pairwise-borda` and `logits-plus-pairwise` on identical seeds. Offline
pairwise accuracy is a prerequisite, never strength evidence. Only a paired
gameplay battery can promote the policy mode.
Both comparator modes are restricted to the incumbent logits' top 16 by
default (`--pairwise-policy-top-k`); do not let a head trained on searched
actions assign support to unseen long-tail actions.

Run `python -m cascadiav3.torch_pairwise_policy_probe` first. It hashes the
checkpoint and every v3 validation shard, requires one source/rules contract,
and reports top-1 plus completed-Q regret for all three policy modes within one
incumbent candidate mask (`--policy-top-k 16`). It records whether that mask
comes from the full menu or a filtered tensor. The confidence gate is applied
to the best two completed-Q actions in the mask; never promote from a filtered
action-surface result.

Current data gate: v3 corrected-rules shards only. The July 9 240-root audit
found 27,360 raw pairs but only 23.33% variance-evaluable; 14.58% of those
clear SNR 1.96. Pre-v3 audit shards cannot become training inputs because
their exact-endgame/provenance fields were not preserved.

### Policy candidate-recall pilot

When a reranker cannot improve decisions inside the retained set, test the
upstream policy projection without perturbing the established Q/value model:

```text
--objective gumbel-selfplay
--policy-head-only
--init-manifest <incumbent>
--selection-metric locked_val_policy
```

The training tensors may be filtered for bounded memory, but the routing gate
may not. `python -m cascadiav3.torch_policy_candidate_probe` requires
unfiltered training-eligible v3 shards and chunk-scores the exact full legal
menu. Cross-host MPS boundary ties may swap a near-equal Kth action, so gate
the baseline with `--min-prior-top-k-overlap 0.99
--require-prior-best-coverage-parity`; exact set equality remains available
for same-numerics audits. A filtered top-K metric is not serving evidence.

If ordinary improved-policy cross-entropy lowers validation loss but does not
improve exact full-menu recall, the only preregistered objective retry is
`--objective gumbel-policy-recall`. Build fixed-width top-64 tensors with
`--filter-mode top-prior-with-q-valid`: retain every Q-valid/selected action,
then fill from incumbent prior rank. The recall loss uses only exact-endgame
roots or roots whose top-two Q labels have count at least 2/action, margin at
least 0.25, and SNR at least 1. It pushes completed-Q best above the 16th
policy logit by 0.25 while a 0.25-weight improved-policy loss limits drift.
Select on `locked_val_policy_confident_best_top16` (max) over the complete
held-out seed block, then rerun the exact full-menu probe. Report the all-root
recall alongside it. Do not add more objective variants if this direct test
fails.

## Checkpoint Contract

Save every 1k optimizer steps and at every epoch/block boundary:

- model weights;
- optimizer, scheduler, scaler;
- RNG states;
- resume-safe loader cursor;
- schema ids;
- dataset manifests and checksums;
- source hashes;
- ruleset ID, exact source revision, exporter artifact identity, and teacher
  manifest/weights identities;
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

Do not promote on loss alone, and do not promote on point deltas alone:
every gate comparison runs at least 100 paired games and requires the paired
delta's 95% confidence interval (`torch_benchmark_stats.paired_delta_stats`,
reported by the benchmark harnesses) to exclude zero. Twenty-game runs are
smoke tests. Search must be public-information-legal (`--rollout-determinize`
for the legacy rollout path; the Gumbel path determinizes by construction).

Gates:

- 95 gate: search-integrated v3 reaches at least 95 mean over at least 100
  matched games without wildlife/category allocation collapse.
- 97 gate: promoted checkpoint beats incumbent/control by at least +0.25 paired
  mean over 250-500 pairs and reaches at least 97 mean.
- 100 gate: freeze champion, run 1,000 all-v3 games, then extend to 4,000 if
  the confidence interval can plausibly cross 100.

Search reports must include:

- exact rules and source revision;
- a complete seed-ordered raw game ledger with per-seat wildlife, habitat,
  Nature Token, and total scores;
- retained count;
- full-search winner retention;
- search regret;
- paired delta;
- decision seconds;
- score breakdown;
- treatment/control timing ratio.

Treatment/control aggregate decision-time ratio above `1.20` is a resource
regression unless explicitly approved before the run.
