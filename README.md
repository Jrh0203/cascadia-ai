# Cascadia AI

This repository contains the Cascadia v3 transformer/search stack and the
exact wildlife-card optimization tools.

## Start here

- [Current status](docs/v3/README.md)
- [Live campaign state](docs/v3/CAMPAIGN_STATE.md)
- [Training pipeline](docs/v3/TRAINING_PIPELINE.md)
- [Machines and runbook](docs/v3/INFRASTRUCTURE.md)
- [Mac mini fleet](docs/v3/FLEET.md)
- [Wildlife catalogs](docs/v3/ALL_WILDLIFE_RULESET_CATALOG.md)

The project uses a lean, strength-first workflow. Runs do not require
preregistration, source or artifact hash verification, receipts, seed
allocation, sealed results, or host-role restrictions. Ordinary configs,
metrics, checkpoints, and direct score comparisons are enough.

Active code:

- `cascadiav3/`: CascadiaFormer, PyTorch training/evaluation, runners, and the
  Rust root exporter;
- `tools/`: exact wildlife solvers, catalog utilities, and fleet helpers;
- `crates/`: Rust rules, simulator, search, data, and model components;
- `python/cascadia_cluster`: optional Bacalhau client;
- `infra/`: optional Bacalhau/MinIO worker configuration.

Large generated datasets, checkpoints, reports, tensor shards, dependencies,
and build outputs are ignored by Git.

## Quick checks

```bash
cargo check --workspace
cargo test --workspace
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src \
  uv run python -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tools \
  uv run pytest -q tests/cluster_unit tools/test_cluster_*.py
```

Pre-v3 material remains recoverable from
`archive/pre-v3-repo-cleanup-2026-07-01`.
