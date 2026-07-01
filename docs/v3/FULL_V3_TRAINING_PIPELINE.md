# Full v3 Implementation And Training Pipeline

This runbook is the operational contract for the additive `cascadiav3/`
expert-root and CascadiaFormer path. It does not replace the legacy
`cascadiav3.pre_gpu.v0` fixtures or the `greedy_policy_tensor_shard_v1`
pretraining corpus.

## Artifact Schemas

- `cascadiav3.pre_gpu.v0`: legacy CPU fixture/search-root schema.
- `greedy_policy_tensor_shard_v1`: compact greedy behavior-cloning tensor shard.
- `cascadiav3.expert_root.v1`: full legal-action expert root JSONL.
- `cascadiav3.expert_tensor_shard.v1`: active packed expert training shard.

Expert roots must reconstruct from `seed_u64` plus `root_replay.replay_prefix`,
must preserve exact legal action order, and must include legal-but-unvisited
actions with `per_action_Q_valid=false` when search does not visit them.

## Packed Tensor Path

The training path writes expert tensors directly from Rust to `.npz`; it does
not materialize full expert JSONL shards. The JSONL exporter remains valuable for
tiny audit fixtures because those records are human-readable and run through the
public-boundary, reconstruction, D6, and category-target validators.

Packed expert shards contain public token features, semantic action features,
per-root offsets, sparse relation edges, selected action labels, Q targets,
Q-valid masks, policy priors, visit statistics, score decomposition labels, and
final score/rank vectors.

Raw all-legal-action expert tensors are audit artifacts, not the default trainer
input. Before training, filter them to a retained top-K action set per root. The
filter keeps the selected action, keeps the best rollout-Q actions up to K,
remaps sparse action relations, writes atomically, and records source checksums.
The current phase0 recovery setting is `K=256`.

After filtering, materialize a fixed-capacity `relation_tail` cache. This keeps
the sparse relation edge list for audit/rebuild while storing the exact
action-row relation suffix the model consumes as a compact `uint8` tensor. The
GPU trainer uses the relation-tail shard, not the sparse-only shard, so the
hot path avoids Python per-edge collation and dense square relation matrices.

The trainer consumes the relation-tail shards with:

```bash
cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter --expert-tensor-corpus --allow-model-fallback --first-seed 2026400000 --seed-count 125 --plies-per-seed 80 --max-actions 32 --rollouts-per-action 1 --rollout-top-k 4 --tensor-compression stored --out cascadiav3/fixtures/full_v3_phase0_bootstrap_tensor_train_tensor.npz --manifest cascadiav3/fixtures/full_v3_phase0_bootstrap_tensor_train_manifest.json
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.expert_tensor_shards --summarize-shard cascadiav3/fixtures/full_v3_phase0_bootstrap_tensor_train_tensor.npz
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.expert_tensor_shards --filter-shard cascadiav3/fixtures/full_v3_phase0_bootstrap_tensor_train_tensor.npz --top-k 256 --out cascadiav3/fixtures/full_v3_phase0_bootstrap_tensor_train_tensor_top256.npz --report cascadiav3/reports/full_v3_phase0_bootstrap_tensor_train_tensor_top256_summary.json
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.expert_tensor_shards --materialize-relation-tail cascadiav3/fixtures/full_v3_phase0_bootstrap_tensor_train_tensor_top256.npz --out cascadiav3/fixtures/full_v3_phase0_bootstrap_tensor_train_tensor_top256_relation_tail.npz --report cascadiav3/reports/full_v3_phase0_bootstrap_tensor_train_tensor_top256_relation_tail_summary.json
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_cascadiaformer --model-size S --train cascadiav3/fixtures/full_v3_phase0_bootstrap_tensor_train_tensor_top256_relation_tail.npz --val cascadiav3/fixtures/full_v3_phase0_bootstrap_tensor_val_tensor_top256_relation_tail.npz --train-format npz --val-format npz --steps 5000 --batch-size 64 --grad-accum 1 --val-max-batches 8 --device cuda
```

## Target Semantics

Primary utility is active-seat raw final score. For each legal action:

```text
per_action_Q = estimated active-seat final raw score
per_action_score_to_go = per_action_Q - exact_afterstate_score_active
```

Losses that consume Q labels must ignore actions with
`per_action_Q_valid=false`. Decomposition labels must sum to each seat's final
score.

## Training Schedule

1. Let the current `john0` greedy pretrain job finish. Fetch only after it is
   complete and only through the existing fetch path.
2. Warm-start `CascadiaFormer-S`:
   - 35k train games, 5k locked validation games.
   - 80 roots/game.
   - Batch 64, gradient accumulation to effective batch 512.
   - AdamW betas `(0.9, 0.95)`, weight decay `0.05`.
   - LR `3e-4`, 2% warmup, cosine decay to 10%.
   - 2 epochs.
3. Search bootstrap S:
   - 100k train roots and 20k locked validation roots.
   - 400 simulations/root.
   - 50k optimizer steps.
   - LR `1e-4`.
4. Scale to `CascadiaFormer-M`:
   - 12 layers, `d_model=768`, 12 heads, FFN 3072.
   - Effective batch 256.
   - LR `5e-5`.
   - 80k optimizer steps.
   - Gradient checkpointing enabled.
5. Expert iteration:
   - 10 cycles, 10k games/cycle.
   - Newest model occupies one rotated focal seat.
   - Opponents are 80% frozen control/champion and 20% prior v3 pool after cycle 1.
   - Label 10k roots/cycle for cycles 1-3.
   - Label 20k roots/cycle for cycles 4-10.
   - Exploration schedule:
     `0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.02`.
   - Train two origins per cycle, three passes at `3e-5`, `3e-5`, `1e-5`.
   - Data mix: 50% current cycle, 30% prior three cycles, 20% bootstrap/older.

## Checkpoint Contract

Save every 1k optimizer steps and every epoch/block boundary. Keep model
weights, optimizer, scheduler, scaler state, RNG state, resume-safe loader
cursor, schema ids, dataset manifests, source hashes, search config, loss
weights, metrics, and model manifest.

Also save the best locked-validation checkpoint and an SWA checkpoint over the
final 20% of optimizer steps.

## Required CPU Gates

```bash
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_schema_registry --include-legacy --include-expert
cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter --chance-mcts-dry-run --allow-model-fallback --seed-count 2 --plies-per-seed 2 --out cascadiav3/fixtures/expert_tiny.jsonl --manifest cascadiav3/fixtures/expert_tiny_manifest.json
cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter --validate-expert-reconstruction --in cascadiav3/fixtures/expert_tiny.jsonl --manifest cascadiav3/fixtures/expert_tiny_manifest.json
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_public_boundary --roots cascadiav3/fixtures/expert_tiny.jsonl --deny-hidden-fields
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_hidden_redetermination --seeds 2026063000:2026063010
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_d6_roundtrip --roots cascadiav3/fixtures/expert_tiny.jsonl
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_category_targets --roots cascadiav3/fixtures/expert_tiny.jsonl
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_cascadiaformer --model-size tiny --train cascadiav3/fixtures/expert_tiny.jsonl --val cascadiav3/fixtures/expert_tiny.jsonl --steps 200 --batch-size 2 --device cpu --overfit-one-batch
cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter --expert-tensor-corpus --allow-model-fallback --first-seed 2026063000 --seed-count 2 --plies-per-seed 2 --rollouts-per-action 1 --rollout-top-k 4 --tensor-compression stored --out cascadiav3/fixtures/expert_tiny_tensor.npz --manifest cascadiav3/fixtures/expert_tiny_tensor_manifest.json
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.expert_tensor_shards --summarize-shard cascadiav3/fixtures/expert_tiny_tensor.npz --report cascadiav3/reports/expert_tiny_tensor_summary.json
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_cascadiaformer --model-size tiny --train cascadiav3/fixtures/expert_tiny_tensor.npz --val cascadiav3/fixtures/expert_tiny_tensor.npz --train-format npz --val-format npz --steps 20 --batch-size 2 --device cuda --val-max-batches 1
```

Promotion cannot be loss-only. Candidate checkpoints must also pass K24/K32
prefilter gates and gameplay gates without regression against the full-search
baseline.
