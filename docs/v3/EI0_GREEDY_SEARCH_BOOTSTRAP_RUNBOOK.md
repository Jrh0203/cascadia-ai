# EI-0 Greedy Search Bootstrap Runbook

Created: 2026-07-01

This is the next v3 strength experiment. It is designed to be resumable after a
lost SSH session, Codex restart, or interrupted run.

## One-Screen Resume Index

If context is lost, start here.

- Source-of-truth doc:
  `docs/v3/EI0_GREEDY_SEARCH_BOOTSTRAP_RUNBOOK.md`.
- Source-of-truth runner:
  `cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh`.
- Benchmark runner:
  `cascadiav3/scripts/run_cascadiaformer_ei0_benchmark_suite.sh`.
- GPU host: `john0` over SSH port `2222`.
- Remote repo root: `/home/john0/cascadia`.
- Remote Python environment: `/home/john0/venvs/torch`.
- Warm start:
  `cascadiav3/checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json`.
- Promotion checkpoint, if training succeeds:
  `cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/guarded_retention_safe_best.manifest.json`.
- Generated tensors, checkpoints, logs, and decision traces are intentionally
  ignored by Git. Persist only source/docs and small summaries.
- Training corpora are packed `.npz` tensors. JSONL is retained only for
  appendable evidence trails such as metrics logs and benchmark decision traces;
  it is not the EI-0 GPU training input.

First command after resuming:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh status
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_benchmark_suite.sh status
git status -sb --untracked-files=all
```

Then follow the checklist below from the first unchecked item. If a prior step
already produced valid artifacts, fetch and audit them rather than recomputing.

## Purpose

The prior corrected greedy-state K32 retention run proved that CascadiaFormer-S
can play near greedy, but it did not beat greedy:

- model mean: `86.7800`
- greedy mean: `87.5875`
- paired delta: `-0.8075`
- exact greedy-action match: `67.3625%`

EI-0 tests whether search-improved labels can improve the model while keeping
training states in greedy's distribution.

## Hypothesis

CascadiaFormer-S can surpass greedy if:

- roots come from greedy self-play states;
- the action menu is greedy-ranked K32;
- the greedy action is always retained at index 0;
- sampled rollout search selects the supervised action when it finds a better
  action;
- training preserves greedy behavior while learning high-confidence deviations.

This is not full expert iteration. Full EI starts only after this bootstrap
passes gameplay gates.

## Runner

Primary runner:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh launch
```

Status, fetch, and stop:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh status
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh fetch
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh stop
```

The wrapper delegates to `run_full_v3_training_pipeline.sh` with:

```text
PROFILE=ei0_greedy_search_bootstrap
EXPERT_TENSOR_MODE=greedy_search_bootstrap
OBJECTIVE=search-improved-greedy-retention
FILTER_MODE=greedy-prefix-strict
FILTER_TOP_K=32
MAX_ACTIONS=32
ROLLOUTS_PER_ACTION=4
ROLLOUT_TOP_K=4
MODEL_SIZE=S
TRAIN_STEPS=25000
BATCH_SIZE=192
LR=0.0001
INIT_MANIFEST=cascadiav3/checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json
```

## Implementation Checklist

Use this as the durable recovery checklist. Mark items in a local note or in the
final handoff, not necessarily by editing this doc during every run.

- [ ] Verify local source state:
  `git status -sb --untracked-files=all`.
- [ ] Verify john0 GPU and warm-start checkpoint are available.
- [ ] Run schema and expert-contract preflight on the synced john0 source.
- [ ] Run tiny EI-0 calibration if machine state, code state, or tensor
  contracts have changed since the last known-good run.
- [ ] Fetch calibration artifacts.
- [ ] Audit calibration tensor invariants:
  selected-action drops must be `0`, max Q invariant error must be `0`, and
  relation-tail tensors must materialize.
- [ ] Launch or resume the full EI-0 greedy search bootstrap run.
- [ ] During generation, confirm CPU exporter processes are active and
  runbook throughput fields eventually become positive.
- [ ] During training, confirm the trainer reaches `25,000` steps or exact
  resume continues from a valid checkpoint manifest.
- [ ] Fetch full EI-0 reports, metrics, logs, checkpoint manifests, and tensor
  summaries.
- [ ] Audit full EI-0 tensor invariants for train and validation shards.
- [ ] Select the promotion checkpoint from
  `guarded_retention_safe_best.manifest.json`, not from the final or SWA
  manifest unless the guard metrics justify that change.
- [ ] Run the no-search complete-game benchmark.
- [ ] Run the search-integrated complete-game benchmark.
- [ ] Fetch benchmark reports, summaries, decision traces, and logs.
- [ ] Update `cascadiav3/EXPERIMENT_LOG.md` with the measured results.
- [ ] Update `docs/v3/PERFORMANCE.md` with changed throughput or gameplay
  facts.
- [ ] Commit only durable source/docs changes. Keep generated `.npz`,
  checkpoints, logs, and decision traces out of Git.

Completion means the run is either promoted, rejected with evidence, or blocked
with exact failing artifacts and next commands. A lower loss alone is not
completion.

## Scale

Default corpus:

- train roots: `250 seeds * 80 plies = 20,000`
- validation roots: `50 seeds * 80 plies = 4,000`
- actions/root: `32`
- rollouts/action: `4`
- total rollout samples: `24,000 * 32 * 4 = 3,072,000`
- tensor format: stored `.npz`
- GPU training input: fixed relation-tail `.npz`

Generated artifacts stay ignored by Git.

## Timeline

Treat generation as the long pole until measured.

1. Preflight and tiny calibration: 30-60 minutes.
2. Full tensor generation: likely overnight on john0 alone.
3. GPU training after tensors exist: roughly 1-3 hours.
4. Benchmark suite: roughly 1-3 hours.

After the first full status report, replace the rough timeline with measured:

- `roots_per_second`
- `rollout_evals_per_second`
- `train_step_seconds`
- train/validation generation seconds

These are written to:

```text
cascadiav3/reports/full_v3_ei0_greedy_search_bootstrap_runbook.json
```

## Preflight Checklist

- [ ] Local tree is clean or all changes are intentional:

```bash
git status -sb --untracked-files=all
```

- [ ] Local main contains the clean v3 repository commit:

```bash
git log -1 --oneline
```

- [ ] john0 is reachable on port 2222:

```bash
ssh -p 2222 john0 'hostname && date -Is && nvidia-smi --query-gpu=index,name,memory.total,memory.used,temperature.gpu --format=csv'
```

- [ ] The greedy-retention warm-start manifest exists on john0:

```bash
ssh -p 2222 john0 'test -s /home/john0/cascadia/cascadiav3/checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json && echo ok'
```

- [ ] CPU/GPU contract gates pass on the synced source. The full runner runs
  these before generation, but if debugging manually use:

```bash
ssh -p 2222 john0 'cd /home/john0/cascadia && . /home/john0/venvs/torch/bin/activate && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.validate_schema_registry --include-legacy --include-expert'
```

## Tiny Calibration Checklist

Run this first if the current machine state is uncertain or if we need a fresh
throughput estimate before committing to the full overnight run.

```bash
SSH_PORT=2222 \
JOB_SLUG=cascadiaformer_ei0_calibration \
PROFILE=ei0_calibration \
TRAIN_SEED_COUNT=4 \
VAL_SEED_COUNT=1 \
PLIES_PER_SEED=20 \
TRAIN_STEPS=500 \
BATCH_SIZE=64 \
EVAL_EVERY_STEPS=100 \
EARLY_STOP_AFTER_STEP=0 \
EARLY_STOP_SELECTION_GUARD_FAILURES=0 \
bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh launch
```

Monitor:

```bash
SSH_PORT=2222 JOB_SLUG=cascadiaformer_ei0_calibration PROFILE=ei0_calibration \
  bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh status
```

Fetch:

```bash
SSH_PORT=2222 JOB_SLUG=cascadiaformer_ei0_calibration PROFILE=ei0_calibration \
  bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh fetch
```

Calibration passes if:

- [ ] train and validation tensors are generated;
- [ ] selected-action dropped count is `0`;
- [ ] relation-tail tensors materialize;
- [ ] trainer reaches step 500;
- [ ] runbook performance JSON reports positive throughput fields.

## Full Launch Checklist

- [ ] No calibration job is still running:

```bash
SSH_PORT=2222 JOB_SLUG=cascadiaformer_ei0_calibration PROFILE=ei0_calibration \
  bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh status
```

- [ ] Launch full EI-0:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh launch
```

- [ ] Record remote log path from launch output. Default:

```text
/home/john0/cascadia/cascadiav3/logs/cascadiaformer_ei0_greedy_search_bootstrap_job.log
```

- [ ] Check status every 30-60 minutes:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh status
```

- [ ] While generation is running, confirm CPU exporter process is visible:

```text
matching expert exporter pid ...
```

- [ ] While training is running, confirm trainer process and GPU usage are
  visible:

```text
matching trainer pid ...
nvidia-smi ...
```

## Expected Artifact Paths

Remote and fetched local paths are the same relative to repo root.

Training tensors:

```text
cascadiav3/fixtures/full_v3_ei0_greedy_search_bootstrap_train_tensor.npz
cascadiav3/fixtures/full_v3_ei0_greedy_search_bootstrap_val_tensor.npz
cascadiav3/fixtures/full_v3_ei0_greedy_search_bootstrap_train_tensor_top32.npz
cascadiav3/fixtures/full_v3_ei0_greedy_search_bootstrap_val_tensor_top32.npz
cascadiav3/fixtures/full_v3_ei0_greedy_search_bootstrap_train_tensor_top32_relation_tail.npz
cascadiav3/fixtures/full_v3_ei0_greedy_search_bootstrap_val_tensor_top32_relation_tail.npz
```

Reports:

```text
cascadiav3/reports/full_v3_ei0_greedy_search_bootstrap_train.json
cascadiav3/reports/full_v3_ei0_greedy_search_bootstrap_metrics.jsonl
cascadiav3/reports/full_v3_ei0_greedy_search_bootstrap_runbook.json
cascadiav3/reports/full_v3_ei0_greedy_search_bootstrap_train_tensor_invariants.json
cascadiav3/reports/full_v3_ei0_greedy_search_bootstrap_val_tensor_invariants.json
cascadiav3/logs/cascadiaformer_ei0_greedy_search_bootstrap_job.log
```

Checkpoints:

```text
cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/
cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/best_locked_val.manifest.json
```

## Resume Checklist

First determine which phase was interrupted.

- [ ] Fetch whatever exists:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh fetch
```

- [ ] Inspect status and last log lines:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh status
```

### If interruption happened during tensor generation

The runner reuses existing tensor files when they are present and non-empty.
Relaunch with the same profile and do not set `REGENERATE_ROOTS=1`:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh launch
```

If a tensor file exists but invariant validation fails, remove only the broken
tensor and its downstream filtered/tail tensors on john0, then relaunch.

### If interruption happened during training

Use exact resume from the latest safe checkpoint manifest. Do not use
`INIT_MANIFEST` and `RESUME_MANIFEST` together.

Example:

```bash
SSH_PORT=2222 \
RESUME_MANIFEST=cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/step_00012000.manifest.json \
bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh launch
```

The runner passes `RESUME_MANIFEST` through to
`torch_train_cascadiaformer --resume`, which restores optimizer, scheduler,
scaler, RNG state, and loader cursor. Resume identity mismatches should fail
closed.

### If interruption happened after training

Fetch artifacts and run the benchmark suite from the selected checkpoint:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh fetch
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_benchmark_suite.sh launch
```

## Benchmark Checklist

Default benchmark suite:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_benchmark_suite.sh launch
```

Monitor:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_benchmark_suite.sh status
```

Fetch:

```bash
SSH_PORT=2222 bash cascadiav3/scripts/run_cascadiaformer_ei0_benchmark_suite.sh fetch
```

Default benchmark outputs:

```text
cascadiav3/reports/cascadiaformer_ei0_no_search_game100.json
cascadiav3/reports/cascadiaformer_ei0_no_search_game100_summary.md
cascadiav3/reports/cascadiaformer_ei0_no_search_game100_decisions.jsonl
cascadiav3/reports/cascadiaformer_ei0_search_game20.json
cascadiav3/reports/cascadiaformer_ei0_search_game20_summary.md
cascadiav3/reports/cascadiaformer_ei0_search_game20_decisions.jsonl
cascadiav3/logs/cascadiaformer_ei0_benchmark_suite_job.log
```

## Success Criteria

Infrastructure success:

- [ ] CPU contract gates pass.
- [ ] Rust exporter tests pass.
- [ ] Tensor invariant validation passes for train and validation shards.
- [ ] Strict K32 selected-action drops are `0`.
- [ ] Relation-tail tensors materialize.
- [ ] Training completes or resumes cleanly to `25,000` steps.
- [ ] Metrics JSONL and runbook JSON are written.
- [ ] Benchmark reports are fetched.

Scientific success:

- [ ] Search labels show non-trivial teacher advantage over greedy.
- [ ] Locked validation total loss decreases.
- [ ] Greedy retention does not collapse.
- [ ] Exact greedy-action match improves over the prior `67.3625%`, or score
  improves despite similar match rate.
- [ ] Q score-to-go metrics improve without invalid-target leakage.

Gameplay success:

- [ ] 100-game no-search model mean is at least greedy minus `0.25`.
- [ ] Strong success: 100-game no-search mean beats greedy by at least `+0.25`.
- [ ] 20-game search-integrated benchmark beats the comparable baseline or
  shows lower regret without category collapse.
- [ ] Treatment/control decision-time ratio is no more than `1.20`, unless a
  slower run was explicitly approved before launch.

Promotion to full expert iteration requires gameplay success, not just lower
loss.

## Failure Triage

If selected-action drops are nonzero:

- stop;
- inspect filter settings;
- verify `FILTER_MODE=greedy-prefix-strict`;
- do not train on the shard.

If greedy retention collapses:

- increase greedy-retention loss weights;
- reduce LR;
- consider longer warm start from greedy-retention checkpoint;
- do not proceed to full expert iteration.

If teacher advantage is weak:

- increase rollout count or improve search teacher;
- avoid scaling model capacity before fixing target quality.

If generation is too slow:

- measure `rollout_evals_per_second`;
- split seed ranges across Bacalhau workers only if outputs are merged through
  durable artifacts and worker disks remain transient;
- keep stored `.npz` for throughput when disk headroom allows.

If GPU utilization is low during training:

- confirm relation-tail tensors exist and are being used;
- inspect batch time and loader throughput in metrics JSONL;
- increase batch only after checking memory headroom.

## Post-Run Checklist

- [ ] Fetch all reports, logs, and checkpoint manifests.
- [ ] Append a concise entry to `cascadiav3/EXPERIMENT_LOG.md`.
- [ ] Update `docs/v3/PERFORMANCE.md` if the run changes measured throughput or
  gameplay facts.
- [ ] Keep generated `.npz`, checkpoints, logs, and decision traces out of Git.
- [ ] Commit only source/docs changes and small summaries that should persist.
