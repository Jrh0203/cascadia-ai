# Cascadia V3 Implementation Package

This directory contains the active transformer implementation:

- schema and validation contracts;
- Python/PyTorch CascadiaFormer model and trainers;
- Rust real-root exporter for greedy and expert tensor shards;
- GPU runner scripts for `john0`;
- tiny fixtures and reports used by tests.

The governing docs are:

- [Rules Contract](../docs/v3/RULES_CONTRACT.md)
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
- `run_gumbel_phase_a_gate.sh` (Gumbel vs honest rollout search, 100 paired games)
- `run_gumbel_ceiling_probe.sh` (512-sim, w=1.0 model-ceiling probe)
- `run_gumbel_selfplay_cycle.sh` (EI-2+ self-play generation + training)
- `run_exact_k1_gate.sh` (fresh same-revision n256/d4 baseline versus exact
  final-personal-turn search, trace-validated before a verdict)
- `run_market_samples_gate.sh` (fresh same-revision n256/d4 market chance-
  sample ablation; reuses the exact-K1 sample-8 control only after validating
  its full rules/source/seeds/search contract)
- `run_model_throughput_probe.sh` (engineering-only M/S/XS/tiny fixed-root
  bridge throughput; measures the compute headroom for model/search inversion)

Gumbel exporter modes (see `--help`):

- `--gumbel-policy-game`: all-seat Gumbel-search games, per-decision JSONL.
- `--gumbel-selfplay-tensor-corpus`: schema-v2 self-play training shards with
  completed-Q targets, improved-policy soft targets, and real-outcome value
  labels.
- `--gumbel-exact-endgame-turns 1`: enumerate the complete legal menu and
  choose by exact own final score on each seat's last personal turn. The
  model and simulations are bypassed for those four decisions; optional
  refresh acceptance is still decided before the hidden replacement draw.
- `python -m cascadiav3.compare_exact_endgame`: compare exact-off/K1 reports
  only after validating rules, source, checkpoint name, seeds, all other
  search settings, exact-decision counts, and identical action traces through
  ply 75. A pre-K1 divergence invalidates the ablation.
- `python -m cascadiav3.compare_market_samples`: compare paired market chance-
  sample counts only after validating rules, source, checkpoint name, seeds,
  all other search settings, sample-count telemetry, and identical action
  traces before the first optional-refresh opportunity. Runs below 100 matched
  games are explicitly engineering smokes, never promotion evidence.
- `python -m cascadiav3.torch_model_throughput_benchmark`: benchmark complete
  collate-to-packed-response throughput for checkpoint and synthetic model
  shapes on identical roots, with hashed model/root provenance and repeated-
  output determinism checks. This is not gameplay evidence.
- `--rollout-determinize`: public-information-legal rollouts for the legacy
  search path (honest baselines).

## Current Status

EI-0 is the first CascadiaFormer run with positive no-search gameplay evidence.
The guarded EI-0 checkpoint scored `89.6175` with q-head serving versus greedy
`87.5575` over 100 complete games.

The rollout-teacher line is retired: one-ply sampled-greedy labeling capped
search strength near `97` (K64/R32 ceiling test), and its serving-time
rollouts observed the true hidden draw order. The active plan is the Gumbel
self-play campaign — see `docs/v3/GUMBEL_SELFPLAY_CAMPAIGN.md`,
`docs/v3/PERFORMANCE.md`, and `cascadiav3/EXPERIMENT_LOG.md`.
