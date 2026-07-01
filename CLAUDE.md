# Cascadia AI

Follow [`AGENTS.md`](AGENTS.md).

The active workstream is Cascadia v3: transformer training, packed expert tensor
generation, GPU serving experiments, and Bacalhau CPU-worker orchestration. Use
`docs/v3/README.md`, `docs/v3/FULL_V3_TRAINING_PIPELINE.md`, and
`cascadiav3/README.md` as the starting points.

Closed v2 research remains available for reproduction, but it should not drive
new v3 architecture decisions without fresh validation. Historical v2 goals,
decision records, and generated reports live in `docs/archive/v2`.

Use:

```bash
cargo check --workspace
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tools uv run pytest -q tests/cluster_unit tools/test_cluster_*.py
```
