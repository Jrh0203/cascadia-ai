# CascadiaFormer EI-0 Overnight Plan

Status: proposed next overnight run.
Date: 2026-07-01.

This plan starts real expert iteration without repeating the distribution-shift
mistake from the first CascadiaFormer expert run. The goal is not to make a raw
no-search transformer instantly average 100. The goal is to create the first
search-improved policy/value checkpoint that can either beat greedy directly or
serve as a stronger prior/value model inside search.

## Current Baseline

The corrected greedy-state CascadiaFormer-S run is the current transformer
baseline:

- Training corpus: 10,000 greedy self-play train roots, 2,000 locked validation
  roots.
- Action menu: strict greedy-ranked K32.
- Objective: pure greedy-retention.
- Locked validation: `locked_val_greedy_top1=0.6780`,
  `locked_val_mean_greedy_rank=2.1640`.
- 100-game complete-game benchmark: CascadiaFormer policy `86.7800`, greedy
  `87.5875`, paired delta `-0.8075`, exact greedy-action match `67.3625%`.

Interpretation: the serving path works and the model is near greedy on score,
but it is not a solved greedy clone. Pure cloning alone cannot reach 100. The
next run needs search-improved labels.

## Objective

Run **EI-0 search bootstrap**:

1. Generate roots from actual greedy self-play states.
2. At each root, retain the greedy-ranked K32 candidate menu.
3. For every retained candidate, estimate active-seat final score by sampled
   greedy rollouts.
4. Train CascadiaFormer-S to prefer the rollout-best action while preserving
   enough greedy retention to avoid policy collapse.
5. Benchmark the resulting no-search policy against greedy.
6. Promote the checkpoint into the next expert-iteration loop only if it clears
   a gameplay gate.

This is expert iteration cycle zero: greedy supplies the state distribution,
search/rollouts supply the improvement operator, and CascadiaFormer distills the
improved action/value targets.

## Required Code Before Launch

Do these before the overnight job starts.

1. Add a greedy-state search-bootstrap exporter mode.

   Current `--expert-tensor-corpus` labels roots and then advances through
   rollout-selected teacher actions. That is useful for later self-play, but it
   recreates the distribution-shift failure if used as the first EI bootstrap.
   We need a mode that:

   - starts each game from greedy self-play,
   - records each greedy-state root before the greedy move is applied,
   - evaluates the root's K32 greedy-ranked candidate menu with rollout labels,
   - writes `cascadiav3.expert_tensor_shard.v1` directly to `.npz`,
   - advances the actual game with the greedy action, not the rollout-best
     teacher action.

   Suggested flag:

   ```bash
   --greedy-state-search-bootstrap-tensor-corpus
   ```

2. Add a runner wrapper for the overnight job.

   Suggested file:

   ```text
   cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh
   ```

   It should reuse `run_full_v3_training_pipeline.sh` machinery but set a new
   `EXPERT_TENSOR_MODE`, profile, objective, and launch parameters.

3. Keep JSONL out of the large path.

   The large artifacts should be stored `.npz` expert tensor shards, filtered
   K32 shards, and relation-tail shards. JSONL remains only for tiny audit
   fixtures.

4. Optional but useful: add trainer warm-start support.

   Warm-starting from
   `cascadiav3/checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json`
   is likely better than training from scratch, but the current trainer does not
   expose an init-manifest argument. If this is not added before launch, start
   from scratch and treat the run as a clean EI-0 baseline.

## Overnight Run Spec

Use a deliberately moderate search budget so the overnight run produces enough
diverse roots rather than a tiny perfect-looking corpus.

### Data

- Train roots: 20,000
- Validation roots: 4,000
- Root source: greedy self-play states
- `plies_per_seed`: 80
- Train seeds: 250
- Validation seeds: 50
- Action menu: K32 greedy-ranked candidates
- Rollouts per action: 4
- Rollout policy: sampled greedy
- Rollout top-k: 4
- Tensor compression: stored
- Filter mode: `greedy-prefix-strict`
- Relation cache: materialized `relation_tail`

Why this size: 20k roots is large enough to cover full-game phase variation
and small enough that K32 x 4 rollouts/root should finish overnight on the 16c/32t
CPU plus RTX 5090 training box.

### Model

- Model: CascadiaFormer-S
- Device: CUDA
- Batch size: 192 or 256
- Gradient accumulation: 1
- Steps: 25,000
- Optimizer: AdamW `(0.9, 0.95)`, weight decay `0.05`
- LR: `2e-4` from scratch, or `1e-4` if warm-starting from greedy-retention
- Warmup: 2%
- Schedule: cosine decay to 10% of starting LR
- SWA: final 20%
- Checkpoint every 1,000 steps and whenever locked-val selection improves

### Loss

Use the existing blended objective as the starting point:

```text
objective = k32-greedy-retention
policy = rollout-best selected action
q = per-action active-seat final score estimate
greedy_policy = legal_actions[0] retention
greedy_margin = keep greedy action competitive
```

Recommended first weights:

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

Selection metric:

```text
locked_val_total, mode=min
```

Also monitor:

- `locked_val_policy`
- `locked_val_q`
- `locked_val_greedy_top1`
- `locked_val_mean_greedy_rank`
- teacher-vs-greedy disagreement rate in the corpus
- mean rollout advantage of selected teacher action over greedy action

## Launch Shape

Once the exporter and runner exist, the intended launch is:

```bash
JOB_SLUG=cascadiaformer_ei0_greedy_search_bootstrap \
PROFILE=ei0_greedy_search_bootstrap \
EXPERT_TENSOR_MODE=greedy_search_bootstrap \
MAX_ACTIONS=32 \
FILTER_TOP_K=32 \
FILTER_MODE=greedy-prefix-strict \
OBJECTIVE=k32-greedy-retention \
SELECTION_METRIC=locked_val_total \
SELECTION_MODE=min \
MODEL_SIZE=S \
TRAIN_FIRST_SEED=2026410000 \
TRAIN_SEED_COUNT=250 \
VAL_FIRST_SEED=2026510000 \
VAL_SEED_COUNT=50 \
PLIES_PER_SEED=80 \
ROLLOUTS_PER_ACTION=4 \
ROLLOUT_TOP_K=4 \
TRAIN_STEPS=25000 \
BATCH_SIZE=192 \
GRAD_ACCUM=1 \
LR=0.0002 \
WEIGHT_DECAY=0.05 \
WARMUP_FRACTION=0.02 \
VAL_MAX_BATCHES=0 \
SWA_FRACTION=0.20 \
bash cascadiav3/scripts/run_full_v3_training_pipeline.sh launch
```

If warm-start support is added, use:

```text
LR=0.0001
INIT_MANIFEST=cascadiav3/checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json
```

## Monitoring During The Run

The job should be allowed to run overnight unless one of these happens:

- exporter crashes or schema validation fails,
- GPU OOM repeats after reducing batch once,
- validation loss is NaN,
- `locked_val_greedy_top1` collapses below `0.20` and stays there after 2,000
  steps,
- runbook report is not produced.

Useful status checks:

```bash
bash cascadiav3/scripts/run_full_v3_training_pipeline.sh status
ssh -p 2222 john0 'tail -n 20 /home/john0/cascadia/cascadiav3/reports/full_v3_ei0_greedy_search_bootstrap_metrics.jsonl'
ssh -p 2222 john0 '/usr/lib/wsl/lib/nvidia-smi'
```

## Morning Evaluation

Fetch artifacts:

```bash
bash cascadiav3/scripts/run_full_v3_training_pipeline.sh fetch
```

Run complete-game benchmarks on `john0`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src \
python -m cascadiav3.torch_cascadiaformer_game_benchmark \
  --manifest cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/best_locked_val.manifest.json \
  --binary cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter \
  --first-seed 2026071000 \
  --games 100 \
  --selection-heads policy \
  --max-actions 32 \
  --baseline-workers 8 \
  --device cuda \
  --out cascadiav3/reports/cascadiaformer_ei0_greedy_search_bootstrap_game100_benchmark.json
```

If the 100-game result is promising, immediately run 500 matched games.

## Decision Gates

### Promote To EI-1

Promote the checkpoint into the next expert-iteration cycle if:

- 100-game paired delta vs greedy is positive, or
- 500-game paired delta vs greedy is at least `+0.25`, or
- no-search policy is neutral but search/MCE using the checkpoint as prior/value
  improves over greedy/search baseline.

### Keep But Do Not Promote

Keep the checkpoint as a supervised bootstrap if:

- direct policy is between `-1.0` and `0.0` versus greedy,
- validation teacher policy and Q losses decrease cleanly,
- selected rollout labels show real advantage over greedy in the corpus.

Next action: more roots or warm-start/weight tuning.

### Reject

Reject as a model-improvement run if:

- direct policy is worse than greedy by more than `1.5`,
- greedy retention collapses and score collapses,
- validation Q/policy losses do not improve,
- teacher labels have little or no measured rollout advantage over greedy.

Next action: improve the teacher/search labels before scaling EI.

## Path To 100

This overnight run is not expected to average 100 directly. It is expected to
answer whether CascadiaFormer can learn a search-improved policy from
greedy-state roots. The path to 100 is:

1. EI-0: greedy-state search bootstrap produces a checkpoint that is neutral or
   positive versus greedy.
2. EI-1/EI-2: generate roots from the promoted checkpoint plus greedy/prior
   pool, using search-improved labels.
3. Scale the model or data only after the promoted S checkpoint improves
   gameplay, not merely validation loss.
4. Use the best checkpoint inside search/MCE, where the value/policy model can
   raise decision quality beyond raw no-search play.
5. Promote only by paired gameplay gates, never by loss alone.
