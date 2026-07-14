# Cascadia AI

This repository is organized around the Cascadia v3 transformer effort:
packed expert-root generation, CascadiaFormer training, GPU/CPU operations, and
Bacalhau-backed worker orchestration for CPU-bound jobs.

## Start Here

- **[Research Pipeline Guide](docs/v3/RESEARCH_PIPELINE_GUIDE.md)** — the
  operator manual: how to read results and run every experiment (screens,
  gates, queues, deploys) end to end with exact commands. Start here to
  RUN things.
- **[Cascadia v3 — Source of Truth](docs/v3/README.md)** — the authoritative
  status of everything; start every session and handoff there.
- **[Research Agenda](docs/v3/RESEARCH_AGENDA.md)** — the living break-100
  experiment queue: priorities, decision rules, and the verdict scoreboard.
- [Transformer architecture](docs/v3/ARCHITECTURE.md)
- [Training pipeline](docs/v3/TRAINING_PIPELINE.md)
- [Rules contract](docs/v3/RULES_CONTRACT.md)
- [Implementation package](cascadiav3/README.md)
- [Bacalhau operations guide](docs/BACALHAU_USAGE.md)
- [Cluster orchestrator notes](docs/cluster_orchestrator.md)
- [Known technical debt](docs/TECH_DEBT.md)

Active code lives in:

- `cascadiav3/`: transformer schemas, PyTorch training/evaluation, run scripts,
  fixtures, and the Rust root exporter.
- `crates/`: current Rust rules, simulator, data, search, evaluation, model,
  and provenance crates.
- `python/cascadia_cluster`: topology-free Bacalhau map/reconnect/artifact
  client.
- `infra/`: Bacalhau, MinIO, and local registry worker configuration.
- `tools/cluster_*.py` plus `tools/r2_map_bacalhau_gate.py`: maintained cluster
  orchestration helpers.

Large generated datasets, checkpoints, reports, tensor shards, dependency
folders, and build outputs are ignored by Git and should be treated as
reproducible or transient run artifacts.

## Historical Recovery

The pre-cleanup v1/v2 archive, old MLX package, legacy teacher bridge, web app,
and rejected experiment attic were removed from `main` to keep this repository
usable. They are recoverable from the Git tag:

```bash
git show archive/pre-v3-repo-cleanup-2026-07-01:<path>
```

## Quick Checks

```bash
cargo check --workspace
cx=cascadiav3/real-root-exporter
cargo test --manifest-path "$cx/Cargo.toml"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tools uv run pytest -q tests/cluster_unit tools/test_cluster_*.py
```
