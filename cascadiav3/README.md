# Cascadia V3 Implementation Package

This directory contains the active transformer implementation:

- schema and validation contracts;
- Python/PyTorch CascadiaFormer model and trainers;
- Rust real-root exporter for greedy and expert tensor shards;
- GPU runner scripts for `john0`;
- tiny fixtures and reports used by tests.

The governing docs are:

- [Architecture](../docs/v3/ARCHITECTURE.md)
- [Training Pipeline](../docs/v3/TRAINING_PIPELINE.md)
- [Operations](../docs/v3/OPERATIONS.md)
- [Performance](../docs/v3/PERFORMANCE.md)

## Layout

- `src/cascadiav3/`: Python package.
- `real-root-exporter/`: Rust exporter for roots and packed tensors.
- `scripts/`: local and `john0` launch/fetch/status wrappers.
- `tests/`: unit tests for schemas, fixtures, replay, tensors, serving
  semantics, and resume contracts.
- `fixtures/`, `reports/`, `checkpoints/`: ignored generated run artifacts.

## Core Commands

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_schema_registry --include-legacy --include-expert
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
```

Tiny audit fixture:

```bash
cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter \
  --chance-mcts-dry-run \
  --allow-model-fallback \
  --seed-count 2 \
  --plies-per-seed 2 \
  --out cascadiav3/fixtures/expert_tiny.jsonl \
  --manifest cascadiav3/fixtures/expert_tiny_manifest.json
```

Packed expert tensor smoke:

```bash
cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter \
  --expert-tensor-corpus \
  --allow-model-fallback \
  --first-seed 2026063000 \
  --seed-count 2 \
  --plies-per-seed 2 \
  --rollouts-per-action 1 \
  --rollout-top-k 4 \
  --tensor-compression stored \
  --out cascadiav3/fixtures/expert_tiny_tensor.npz \
  --manifest cascadiav3/fixtures/expert_tiny_tensor_manifest.json
```

## Runner Pattern

Long-running scripts generally support:

```bash
bash cascadiav3/scripts/<runner>.sh launch
bash cascadiav3/scripts/<runner>.sh status
bash cascadiav3/scripts/<runner>.sh fetch
bash cascadiav3/scripts/<runner>.sh stop
```

Important runners:

- `run_john0_gpu_smoke.sh`
- `run_greedy_policy_pretrain.sh`
- `run_cascadiaformer_greedy_k32_retention.sh`
- `run_full_v3_training_pipeline.sh`

## Current Status

EI-0 is the first CascadiaFormer run with positive no-search gameplay evidence.
The guarded EI-0 checkpoint scored `89.6175` with q-head serving versus greedy
`87.5575` over 100 complete games.

The search-integrated gate is strong but not promoted over full search:
CascadiaFormer-search K32 of K64 scored `95.8000` versus matched full K64 search
at `96.9750` over 20 games. See `docs/v3/PERFORMANCE.md` and
`cascadiav3/EXPERIMENT_LOG.md` before choosing the next scale-up.
