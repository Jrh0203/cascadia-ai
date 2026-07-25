# CascadiaFormer Training Pipeline

## Objective

Train a policy/value model that improves actual four-player Cascadia score.
Offline metrics diagnose the model; complete games decide whether an
iteration helped.

## Correctness boundary

The Rust game engine is the authority for legal actions, chance transitions,
and scoring. Keep these engineering invariants:

- legal-action coverage is complete;
- selected actions survive filtering;
- public model inputs do not expose hidden future draws;
- spatial transforms round-trip;
- category targets sum to the scalar score;
- serving ranks
  `exact_afterstate_score_active + predicted_score_to_go`;
- non-finite losses or gradients abort the update.

These are behavior checks, not provenance gates.

## Data

Use packed `.npz` shards for real training:

- `greedy_policy_tensor_shard_v1` for behavior cloning;
- `cascadiav3.expert_tensor_shard.v1` for packed expert roots;
- v2 for improved-policy and search-root-value targets;
- v3 for explicit exact-endgame rows;
- v4 for active-seat and structured wildlife/habitat/Nature targets.

New work should normally use v4. Older readable shards may be mixed whenever
their fields satisfy the selected objective. Source revision, artifact hashes,
teacher hashes, rules IDs, scientific eligibility labels, and disjoint seed
ranges are not admission requirements.

Large corpora should be memory-mapped. JSONL is useful for small diagnostics.

## Target semantics

For each legal action:

```text
completed_Q = estimated active-seat final score
score_to_go = completed_Q - exact_afterstate_score_active
```

For structured Q:

```text
category_score_to_go =
    terminal_category_score(active_seat)
    - exact_selected_afterstate_category_score(active_seat)
```

Only the selected trajectory has a terminal category decomposition. Do not
invent category labels for unselected counterfactual actions.

## Bootstrap

Greedy behavior cloning is a convenient initialization:

- generate greedy self-play states;
- retain K32 or K64 legal menus;
- train policy and score-to-go heads;
- validate in complete games.

This stage is optional when a usable checkpoint already exists.

## Expert iteration

The normal loop is Gumbel self-play:

1. all four seats play from the current checkpoint;
2. Gumbel search supplies completed-Q and improved-policy targets;
3. real terminal outcomes supply value/rank/category labels;
4. train a warm-started model on recent and replay data;
5. evaluate the new checkpoint immediately;
6. continue from whichever checkpoint plays better.

Existing entry point:

```bash
MODEL_MANIFEST=<checkpoint.manifest.json> \
PROFILE=<run-name> \
bash cascadiav3/scripts/run_gumbel_selfplay_cycle.sh run
```

Important scale knobs:

```text
TRAIN_SEED_COUNT
VAL_SEED_COUNT
PLIES_PER_SEED
GUMBEL_N_SIMULATIONS
GUMBEL_DETERMINIZATIONS
MODEL_SESSIONS
RAYON_THREADS
TRAIN_STEPS
BATCH_SIZE
LR
TRAIN_SOURCE_WEIGHTS
```

Tune them from measured end-to-end label quality, training throughput, GPU
utilization, and gameplay. The old 1,250/125-seed, n64/d4, 25k-step recipe is
a starting point, not a limit.

## Model selection

Write:

- numbered checkpoints;
- `latest`;
- `best` by the chosen validation metric;
- a metrics JSONL file;
- the run configuration.

Resume if the checkpoint loads into the intended architecture. Do not block
resume on source hashes, dataset hashes, exact launch identity, or seed
history. A changed dataset or hyperparameter is simply a new phase of the
same run if that is operationally useful.

Evaluate candidates on visible game batches. Paired seeds reduce variance;
confidence intervals communicate uncertainty. Neither is a mandatory
promotion ceremony. Start small, watch results, and add games while the
decision is genuinely uncertain.

## Performance

- Prefer one resident batched inference service to many CUDA contexts.
- Saturate the GPU with concurrent independent games or roots.
- Benchmark TF32, bf16, compilation, batch size, worker count, and bridge
  concurrency on the current hardware.
- Use john1–john4 for any generation, search, or evaluation they accelerate.
- Keep enough telemetry to identify the bottleneck, then remove telemetry that
  costs more than it teaches.

## Validation

Run focused tests while iterating and the relevant broad suites before
shipping engine/model-format changes:

```bash
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src \
  uv run python -m unittest discover -s cascadiav3/tests -v
```
