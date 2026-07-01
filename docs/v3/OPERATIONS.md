# V3 Operations

## Local Checks

```bash
make check
make test
```

Equivalent explicit commands:

```bash
cargo check --workspace
cargo test --workspace
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tools uv run pytest -q tests/cluster_unit tools/test_cluster_*.py
```

## Artifact Policy

Keep generated artifacts out of Git:

- tensor shards: `.npz`, `.npy`;
- checkpoints: `.pt`, `.safetensors`;
- large reports and decision traces;
- local dependency folders;
- Cargo and real-root-exporter build outputs.

Worker-local Bacalhau disk is scratch only. Durable artifacts must be exported
through S3Managed `/outputs`, imported into the orchestrator artifact root, and
removed from workers after completion.

## john0 GPU Runners

Use `cascadiav3/scripts` as the operational entry point. Most long jobs expose:

```bash
bash cascadiav3/scripts/<runner>.sh launch
bash cascadiav3/scripts/<runner>.sh status
bash cascadiav3/scripts/<runner>.sh fetch
bash cascadiav3/scripts/<runner>.sh stop
```

Important runners:

```bash
bash cascadiav3/scripts/run_john0_gpu_smoke.sh
bash cascadiav3/scripts/run_greedy_policy_pretrain.sh status
bash cascadiav3/scripts/run_cascadiaformer_greedy_k32_retention.sh status
bash cascadiav3/scripts/run_full_v3_training_pipeline.sh status
```

Real training data should stay in packed tensors:

```bash
CORPUS_FORMAT=npz TENSOR_EXPORTER=rust TENSOR_COMPRESSION=stored KEEP_JSONL=0 \
  bash cascadiav3/scripts/run_greedy_policy_pretrain.sh launch
```

## Bacalhau CPU Workers

Use the cluster package and `docs/BACALHAU_USAGE.md` for worker scheduling. The
cluster client is topology-free: callers describe independent work items and
resource requirements, not host placement.

Current intended shape:

- john1: orchestrator plus constrained worker capacity;
- john2/john3/john4: dedicated CPU workers;
- worker disk: transient;
- outputs: S3Managed plus manifest/receipt validation;
- retries: through durable request IDs, not manual host reassignment.

## Recovery

- Reconnect Bacalhau jobs by request ID.
- Resume training only from a matching manifest and source identity.
- Do not reuse deleted local worker artifacts as durable evidence.
- If a long run is interrupted, fetch manifests and metrics before deciding
  whether to resume, restart, or discard.
