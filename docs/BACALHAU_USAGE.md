# Bacalhau Usage Runbook

This is the operational reference for scheduling Cascadia CPU jobs on the
private Bacalhau fabric. It is written for future agents working in this
repository: if the rest of the thread context is gone, this file should still be
enough to build an image, submit work, retrieve artifacts, and understand the
storage lifecycle.

## Mental Model

Bacalhau is the scheduler. John1 is the orchestrator, private image registry,
private object store, and a small compute node. John2, john3, and john4 are
dedicated compute nodes.

The caller submits independent container jobs. Bacalhau chooses placement. Do
not hand-shard by host, SSH to workers for artifacts, or write logic that depends
on a physical machine unless the task is an explicit infrastructure smoke test.

Worker disk is transient scratch. Durable artifacts must be published through
Bacalhau result paths and imported from MinIO. Anything left on a worker
filesystem after the job exits can disappear.

## Endpoints And Roots

| Purpose | Endpoint or path |
|---|---|
| Bacalhau API | `http://100.110.109.6:1234` |
| Bacalhau Web UI | `http://100.110.109.6:8438` |
| NATS control plane | `nats://100.110.109.6:4222` |
| Private OCI registry | `100.110.109.6:5000` |
| MinIO API | `http://100.110.109.6:9000` |
| John1 runtime root | `/Users/johnherrick/cascadia-bench/orchestrator` |
| Worker runtime roots | `/Users/johnN/cascadia-cluster` on john2-john4 |

Useful local environment:

```bash
export BACALHAU_API_HOST=100.110.109.6
export BACALHAU_API_PORT=1234
export DOCKER_HOST=unix:///Users/johnherrick/.local/share/cascadia-r2/colima/cascadia-r2/docker.sock
```

The pinned Bacalhau CLI is:

```bash
/Users/johnherrick/cascadia-bench/orchestrator/bin/bacalhau
```

## Capacity Contract

Live intended capacity:

| Node | CPU | Memory | Disk |
|---|---:|---:|---:|
| john1 | 9 | 12 GiB | 80 GiB transient |
| john2 | 10 | 15 GiB | 80 GiB transient |
| john3 | 10 | 15 GiB | 80 GiB transient |
| john4 | 10 | 15 GiB | 80 GiB transient |

Total scheduler capacity is 39 CPUs, 57 GiB memory, and 320 GiB transient disk.

Disk requests are admission controls for execution scratch. They are not a
durable-storage reservation. Large output jobs should write final artifacts
under `/outputs`; Bacalhau publishes that directory to MinIO and the client
imports it into a canonical local artifact directory.

## Health Checks

Run this before any real campaign:

```bash
PYTHONPATH=python:tools uv run python tools/cluster_fabric_health.py
```

Healthy means:

- exactly john1, john2, john3, and john4 are present;
- every node is connected;
- every node advertises Docker support;
- every node runs Bacalhau `v1.9.0`;
- CPU, memory, and disk match the contract above;
- john1 registry and MinIO health endpoints respond.

Raw scheduler view:

```bash
BACALHAU_API_HOST=100.110.109.6 BACALHAU_API_PORT=1234 \
  /Users/johnherrick/cascadia-bench/orchestrator/bin/bacalhau node list --output json |
  jq '[.[] | {
    name: .Info.Labels.cascadia_internal_node,
    connection: .Connection,
    engines: .Info.ComputeNodeInfo.ExecutionEngines,
    capacity: .Info.ComputeNodeInfo.MaxCapacity,
    available: .Info.ComputeNodeInfo.AvailableCapacity,
    running: .Info.ComputeNodeInfo.RunningExecutions
  }]'
```

## Storage Startup

Storage is separate from Bacalhau scheduler startup. Start or repair registry
and MinIO on john1 with:

```bash
make cluster-fabric-start-storage
```

This runs:

```bash
CASCADIA_CLUSTER_ROOT=/Users/johnherrick/cascadia-bench/orchestrator \
DOCKER_HOST=unix:///Users/johnherrick/.local/share/cascadia-r2/colima/cascadia-r2/docker.sock \
/Users/johnherrick/cascadia-bench/orchestrator/bin/run-storage.zsh
```

Registry and MinIO container data are host bind mounts:

- registry: `/Users/johnherrick/cascadia-bench/orchestrator/state/registry`
- MinIO: `/Users/johnherrick/cascadia-bench/orchestrator/state/minio`

This means Colima/Docker scratch can be reset without deleting object-store or
registry state, as long as storage containers are recreated afterward.

## Build And Publish Images

Only build research images on john1. Use immutable digests only; mutable tags are
rejected by the Python API.

```bash
uv run python tools/cluster_build_push.py \
  --context . \
  --dockerfile Dockerfile \
  --name worker-name \
  --tag descriptive-tag \
  --receipt artifacts/cluster/images/worker-name-descriptive-tag.json
```

The receipt contains `image_digest`, for example:

```text
100.110.109.6:5000/cascadia/worker-name@sha256:<64 hex chars>
```

Use that digest in every job. Do not submit `:latest` or any mutable tag.

Research images should include `/usr/local/bin/cascadia-cluster-job` and set it
as the entrypoint. That wrapper:

- verifies content-addressed inputs before the command runs;
- materializes model bundles under `/tmp/cascadia-models`;
- creates a transient `$CASCADIA_SCRATCH_ROOT`;
- writes `/outputs/manifest.json`;
- records deterministic failures as published failure manifests;
- preserves retryable runtime exits for Bacalhau recovery;
- removes `$CASCADIA_SCRATCH_ROOT` and `/tmp/cascadia-models` on exit.

## Preferred Python Submission Path

Use `ClusterClient` for real work. It provides request persistence, immutable
spec validation, content-addressed inputs, S3Managed results, manifest
validation, and canonical artifact import.

```python
from pathlib import Path

from cascadia_cluster import (
    ClusterClient,
    ContainerInput,
    ObjectStoreClient,
    ObjectStoreConfig,
    Resources,
)

root = Path("/Users/johnherrick/cascadia-bench/orchestrator")
secrets = {}
for line in (root / "config/secrets.env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        key, value = line.split("=", 1)
        secrets[key] = value

object_store = ObjectStoreClient(
    ObjectStoreConfig(
        endpoint="http://100.110.109.6:9000",
        access_key=secrets["MINIO_ROOT_USER"],
        secret_key=secrets["MINIO_ROOT_PASSWORD"],
    )
)
object_store.ensure_bucket("cascadia-inputs")
object_store.ensure_bucket("cascadia-results")

cluster = ClusterClient(
    "http://100.110.109.6:1234",
    state_directory=Path("artifacts/cluster/requests"),
    object_store=object_store,
    artifact_directory=Path("artifacts/cluster/accepted"),
)

image = "100.110.109.6:5000/cascadia/worker@sha256:<digest>"

results = cluster.map(
    image=image,
    jobs=[
        ContainerInput("seed-0000", args=("simulate", "--seed", "0")),
        ContainerInput("seed-0001", args=("simulate", "--seed", "1")),
    ],
    resources=Resources(cpu=2, memory_gib=4, disk_gib=4),
    outputs=("/outputs",),
    timeout_seconds=3600,
    entrypoint=("/usr/local/bin/cascadia-cluster-job",),
    experiment_id="example-v1",
    scheduler_backpressure=True,
)

for result in results:
    print(result.item_key, result.status, result.artifact_manifest)
```

Use `submit_map` when you want a reconnectable handle:

```python
handle = cluster.submit_map(
    image=image,
    jobs=jobs,
    resources=Resources(cpu=2, memory_gib=4, disk_gib=4),
    outputs=("/outputs",),
    timeout_seconds=3600,
    entrypoint=("/usr/local/bin/cascadia-cluster-job",),
    experiment_id="long-run-v1",
    request_id="long-run-v1-20260701",
    scheduler_backpressure=True,
)

handle.wait(timeout_seconds=3600)
summary = handle.results()
```

Keep `request_id` stable for resumable long runs. Reusing the same request ID
with the same spec reconnects. Reusing it with a different spec fails closed.

## Inputs

Inputs should be staged into MinIO by SHA-256. Each `InputReference` mounts one
object under a target directory. The entrypoint validates the file digest before
the application command sees it.

The public model is:

```python
InputReference(
    bucket="cascadia-inputs",
    key="sha256/ab/abcdef.../input.bin",
    sha256="abcdef...",
    target="/inputs/input-0000",
    endpoint="http://100.110.109.6:9000",
)
```

The mounted file path is `target / basename(key)`. Use
`reference.mounted_path` when constructing command arguments.

Do not pass secrets in `ContainerInput.environment`; secret-looking environment
keys are rejected by the API.

## Outputs And Artifact Lifecycle

Inside the container:

- write durable result files under `/outputs`;
- write temporary work under `$CASCADIA_SCRATCH_ROOT`;
- do not write final artifacts to `/tmp`, `/var/tmp`, the working directory, or
  any other worker-local path;
- keep output paths relative, normal, and symlink-free;
- let `/usr/local/bin/cascadia-cluster-job` create `manifest.json`.

After the command exits:

1. the entrypoint writes `/outputs/manifest.json`;
2. Bacalhau S3Managed publishes `/outputs` to MinIO under
   `cascadia-results/executions/<job-id>/<execution-id>.tar.gz`;
3. `ClusterClient` downloads, validates, and atomically imports that result into
   the configured `artifact_directory`;
4. worker-local scratch is disposable and must not be used as a source of truth.

The importer rejects traversal, links, undeclared files, size mismatches, and
checksum mismatches.

## Resource Requests

Request what the job really needs:

```python
Resources(cpu=2, memory_gib=4, disk_gib=8)
```

Guidelines:

- CPU is whole or fractional cores, not millicores.
- Memory is GiB.
- Disk is transient scratch GiB and must be <= 80 on this fabric.
- `scheduler_backpressure=True` is recommended for large maps; it admits only
  as much work as the current connected capacity can pack.
- Do not lower resource requests to make an oversized job schedule. Split the
  job or reduce its actual working set.

## CLI Smoke Jobs

Use raw Bacalhau CLI for infrastructure probes, not for canonical research
artifacts.

Pinned-node smoke:

```bash
BACALHAU_API_HOST=100.110.109.6 BACALHAU_API_PORT=1234 \
/Users/johnherrick/cascadia-bench/orchestrator/bin/bacalhau docker run \
  --name john2-smoke-$(date +%Y%m%d%H%M%S) \
  --constraints cascadia_internal_node=john2 \
  --cpu 100m --memory 128Mb --disk 1Gb \
  --timeout 120 --wait-timeout-secs 300 \
  alpine:3.20 -- sh -lc 'echo OK && uname -m && df -h /'
```

General smoke without host pinning:

```bash
BACALHAU_API_HOST=100.110.109.6 BACALHAU_API_PORT=1234 \
/Users/johnherrick/cascadia-bench/orchestrator/bin/bacalhau docker run \
  --name fabric-smoke-$(date +%Y%m%d%H%M%S) \
  --cpu 100m --memory 128Mb --disk 1Gb \
  --timeout 120 --wait-timeout-secs 300 \
  alpine:3.20 -- sh -lc 'echo OK && hostname'
```

Inspect a job:

```bash
JOB=j-...
BACALHAU_API_HOST=100.110.109.6 BACALHAU_API_PORT=1234 \
  /Users/johnherrick/cascadia-bench/orchestrator/bin/bacalhau job describe "$JOB"

BACALHAU_API_HOST=100.110.109.6 BACALHAU_API_PORT=1234 \
  /Users/johnherrick/cascadia-bench/orchestrator/bin/bacalhau job executions "$JOB" --output json
```

## Operational Commands

Install or refresh the fabric:

```bash
make cluster-fabric-install
```

Run health:

```bash
make cluster-fabric-health
```

Run unit and integration fabric tests:

```bash
make cluster-fabric-test
```

Start storage:

```bash
make cluster-fabric-start-storage
```

Start or restart remote fallback workers manually when remote launchd bootstrap
is unavailable:

```bash
ssh john2 'ROOT=/Users/john2/cascadia-cluster; nohup env \
  HOME=/Users/john2 \
  PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  CASCADIA_CLUSTER_ROOT="$ROOT" \
  CASCADIA_BACALHAU_ROLE=compute \
  DOCKER_HOST="unix:///Users/john2/.colima/default/docker.sock" \
  "$ROOT/bin/run-forever.zsh" \
  >"$ROOT/logs/bacalhau-supervisor.stdout.log" \
  2>"$ROOT/logs/bacalhau-supervisor.stderr.log" </dev/null &'
```

Prefer `tools/cluster_fabric_install.py` over hand-starting unless actively
repairing a node.

## Troubleshooting

Worker connected but no Docker engine:

- Verify `DOCKER_HOST` points at the Colima socket.
- Restart the worker with the fallback supervisor command above.
- Re-run `tools/cluster_fabric_health.py`.

Worker disconnected after john1 restart:

- Restart Bacalhau on the worker; the Colima VM usually does not need to move.
- Wait for the heartbeat interval and re-run health.

Job queued forever:

- Check requested CPU, memory, and disk against live node capacity.
- Check image digest reachability from workers.
- Check whether `scheduler_backpressure=True` should be used for a large map.

No artifact imported:

- Inspect `job executions`.
- Confirm the image used `/usr/local/bin/cascadia-cluster-job`.
- Confirm the application wrote files under `/outputs`.
- Confirm `manifest.json` exists in the published output.
- Confirm MinIO buckets exist and credentials came from john1
  `config/secrets.env`.

Do not recover artifacts by SSH-ing into worker scratch directories. If it was
not published and imported, it is not a durable artifact.

## Cleanup Policy

Allowed and expected:

- reset Colima Docker data disks when no executions are running;
- prune worker Docker images and containers;
- remove worker-local execution scratch after jobs finish;
- rotate Bacalhau logs;
- prune old MinIO inputs only when the associated request/artifact provenance is
  no longer needed.

Not allowed without a deliberate campaign decision:

- deleting john1 `state/minio` or `state/registry`;
- deleting canonical accepted artifacts;
- deleting campaign state files used for reproducibility;
- relying on worker-local files as final outputs.

Before disk maintenance, verify idleness:

```bash
BACALHAU_API_HOST=100.110.109.6 BACALHAU_API_PORT=1234 \
/Users/johnherrick/cascadia-bench/orchestrator/bin/bacalhau node list --output json |
jq '[.[] | {
  name: .Info.Labels.cascadia_internal_node,
  running: .Info.ComputeNodeInfo.RunningExecutions,
  enqueued: .Info.ComputeNodeInfo.EnqueuedExecutions
}]'
```

All values should be zero before resetting Colima scratch.
