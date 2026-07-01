# Cascadia AI

Follow [`AGENTS.md`](AGENTS.md).

The active workstream is Cascadia v3: transformer training, packed expert tensor
generation, GPU serving experiments, and Bacalhau CPU-worker orchestration. Use
`docs/v3/README.md`, `docs/v3/ARCHITECTURE.md`,
`docs/v3/TRAINING_PIPELINE.md`, and `cascadiav3/README.md` as the starting
points.

Historical v1/v2 material was intentionally removed from `main` during the
2026-07-01 cleanup. It remains recoverable from
`archive/pre-v3-repo-cleanup-2026-07-01` when reproduction is truly needed, but
it should not drive v3 architecture decisions without fresh validation.

Use:

```bash
cargo check --workspace
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tools uv run pytest -q tests/cluster_unit tools/test_cluster_*.py
```
