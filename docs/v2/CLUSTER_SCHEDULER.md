# Cluster Research Scheduler

The active scheduler is the private Bacalhau v1.9.0 execution fabric described
in [`../cluster_orchestrator.md`](../cluster_orchestrator.md). The former
host-aware manifest queue is frozen historical evidence and is not a workload
submission path.

## Authority

- Bacalhau owns node selection, admission, queueing, attempts, rescheduling,
  cancellation, and live job state.
- `python/cascadia_cluster` owns typed topology-free submission, idempotency,
  reconnect, ordered results, and artifact validation.
- The experiment ledger owns hypotheses, protocols, metrics, and conclusions.
- John1 owns source, image publication, active artifacts, and aggregation.

Do not mirror live Bacalhau jobs into `research-queue-v1.json`.

## Public API

```python
from cascadia_cluster import ClusterClient, ContainerInput, Resources

results = cluster.map(
    image="100.110.109.6:5000/cascadia/worker@sha256:...",
    jobs=[
        ContainerInput("shard-000", args=("simulate", "--seed-start", "0")),
        ContainerInput("shard-001", args=("simulate", "--seed-start", "10000")),
    ],
    resources=Resources(cpu=4, memory_gib=6, disk_gib=8),
    outputs=("/outputs",),
    timeout_seconds=3600,
)
```

For long work, use `submit_map`, persist the request ID, and reconnect with
`ClusterClient.reconnect`. A lost caller does not cancel running work.

Public requests reject mutable image tags, host/node/SSH/remote-root fields,
secret-like environment keys, duplicate item keys, malformed resources, and
non-content-addressed inputs. Results retain submitted item order regardless of
completion order.

## Fabric

- Orchestrator/API: `http://100.110.109.6:1234`
- Web UI: `http://100.110.109.6:8438`
- Registry: `100.110.109.6:5000`
- MinIO: `http://100.110.109.6:9000`
- Compute: john1, john2, john3
- Excluded from scheduling: john4

John1 builds and pushes each `linux/arm64` image once. Jobs reference the
resolved digest. Workers pull and cache layers; no `docker save`, `rsync`,
`docker load`, remote source checkout, or worker build exists in the active
path.

## Inputs and outputs

Large inputs are staged once in `cascadia-inputs` under keys derived from
SHA-256. Each execution publishes a result archive into `cascadia-results`
under its job and execution IDs.

The worker entrypoint writes `cascadia.cluster.output-manifest.v1`, declaring
every path, byte count, SHA-256, command, protocol, and application metadata.
John1 streams the archive, rejects unsafe tar members and undeclared bytes,
validates all checksums, and atomically imports one accepted execution. A
duplicate success remains diagnostic evidence and cannot overwrite the accepted
artifact.

## Failure behavior

- Deterministic application failures are written as validated failure
  manifests and execute once.
- Transient exits remain nonzero so Bacalhau can reschedule within the
  fabric-wide three-attempt bound.
- Unschedulable resource requests remain visible; resources are never silently
  reduced.
- Partial map failure preserves successful sibling results in `MapError`.
- Cancellation stops only nonterminal jobs.
- Scheduler and caller restart reconnect to the same job/specification hashes.

## Operations

```bash
make cluster-fabric-install
make cluster-fabric-health
make cluster-fabric-test
make cluster-fabric-start-storage
make cluster-fabric-build-push CLUSTER_IMAGE_TAG=dev
make cluster-fabric-canary
```

The dashboard at `http://100.110.109.6:5187/cluster` shows john1-john4 fleet
telemetry, john1 MLX state, Bacalhau totals, john1-john3 allocation, service
health, recent jobs, attempts, and failure reasons.

## Frozen legacy queue

The cutover snapshot is:

```text
artifacts/cluster/legacy-freeze/bacalhau-cutover-20260619-v1/manifest.json
```

`research-queue-v1.json` and its SQLite/attempt history remain readable for
provenance. Mutation commands fail closed unless an explicit rollback override
is supplied. Do not delete, rewrite, or use those records as Bacalhau state.
