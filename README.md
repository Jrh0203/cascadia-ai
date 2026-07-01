# Cascadia AI

This repository is now organized around the Cascadia v3 transformer effort:
packed expert-root generation, CascadiaFormer training, GPU/CPU operations, and
Bacalhau-backed worker orchestration for CPU-bound jobs.

## Current Work

- [Cascadia v3 overview](docs/v3/README.md)
- [Cascadia v3 implementation package](cascadiav3/README.md)
- [Full v3 training pipeline](docs/v3/FULL_V3_TRAINING_PIPELINE.md)
- [Bacalhau operations guide](docs/BACALHAU_USAGE.md)
- [Cluster orchestrator notes](docs/cluster_orchestrator.md)
- [Known technical debt](docs/TECH_DEBT.md)

The active v3 code lives in `cascadiav3/`, `infra/`, `python/cascadia_cluster`,
`tests/cluster_unit`, and the `tools/cluster_*.py` / `tools/v3_*.py` helpers.
Large generated datasets, checkpoints, reports, and tensor shards are ignored
by Git and should be treated as reproducible or transient run artifacts.

## Historical Systems

- Cascadia v2 is closed at a 95.744 mean base-score final validation. Its
  maintained source remains in `crates/`, `python/cascadia_mlx`, `apps/web`,
  and `docs/v2`.
- Bulky v2 decision logs and generated evidence reports are archived under
  `docs/archive/v2`.
- The original v1 implementation is retained under `legacy/` for reference and
  differential checks. It is intentionally excluded from default workspace
  membership.

## Quick Checks

```bash
cargo check --workspace
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tools uv run pytest -q tests/cluster_unit tools/test_cluster_*.py
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
```

Use v2-specific Make targets only when reproducing or auditing the archived v2
system. New strength work should start from the v3 runbooks above.
