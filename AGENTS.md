# Cascadia AI

## Goal

Build a superhuman Cascadia board-game AI, currently through the Cascadia v3
transformer stack.

## Standards

- Boil the ocean: ship the complete fix with tests and documentation when it is
  within reach.
- Search before building. Test before shipping.
- No hacks or undocumented shortcuts. If an unavoidable compromise remains,
  document it in `docs/TECH_DEBT.md` with cause, proper fix, and blast radius.

## Current Source Of Truth

- Read `docs/v3/README.md` first.
- Architecture lives in `docs/v3/ARCHITECTURE.md`.
- Training and promotion methodology lives in `docs/v3/TRAINING_PIPELINE.md`.
- Implementation details and command entry points live in `cascadiav3/README.md`.
- Bacalhau worker operation lives in `docs/BACALHAU_USAGE.md`.

The pre-cleanup v1/v2 archive, legacy teacher bridge, old MLX package, web app,
and rejected experiment archive are intentionally not on `main`. Recover them
from `archive/pre-v3-repo-cleanup-2026-07-01` only when reproducing historical
evidence is explicitly required.

## Engineering Rules

- Keep generated data, checkpoints, reports, dependency directories, and build
  outputs out of Git.
- Prefer packed tensor `.npz` paths for real training data. Keep JSONL only for
  tiny audit fixtures.
- Radius 6 is the default public board fast path for CascadiaFormer. Exact
  overflow is required for states outside the disk.
- Do not claim strength from validation loss alone. Promotion needs paired
  gameplay evidence and score/category breakdowns.
- Before shipping code, run the relevant subset of:

```bash
cargo check --workspace
cargo test --workspace
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tools uv run pytest -q tests/cluster_unit tools/test_cluster_*.py
```
