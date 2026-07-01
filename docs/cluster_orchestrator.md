# Cascadia container execution fabric

## Contract

New CPU batch work runs through Bacalhau v1.9.0. Research callers submit an
immutable image plus independent `ContainerInput` values and never name a machine.
Bacalhau places and recovers work across john1, john2, john3, and john4. Native
MLX training stays outside this fabric on john1. Campaign-specific gates may
still narrow the eligible set when an older experiment preregistered a smaller
fabric.

The old manifest queue is frozen at
`artifacts/cluster/legacy-queue-freeze-v1.json`. Its records remain readable evidence;
all mutating commands fail closed unless an explicit rollback override is supplied.

## Runtime topology

| Service | Private endpoint | Durable state |
|---|---|---|
| Bacalhau REST API | `http://100.110.109.6:1234` | john1 orchestration root |
| Bacalhau Web UI | `http://100.110.109.6:8438` | Bacalhau API |
| NATS control plane | `nats://100.110.109.6:4222` | Bacalhau state |
| OCI registry | `http://100.110.109.6:5000` | john1 `state/registry` |
| MinIO | `http://100.110.109.6:9000` | john1 `state/minio` |

The API and storage endpoints bind to john1's Tailscale address, not its public
interfaces. Compute membership authentication and MinIO credentials are generated
under the runtime root with mode `0600`; they are not committed.

## Public Python API

```python
from pathlib import Path

from cascadia_cluster import ClusterClient, ContainerInput, Resources

cluster = ClusterClient(
    "http://100.110.109.6:1234",
    state_directory=Path("artifacts/cluster/requests"),
)
results = cluster.map(
    image="100.110.109.6:5000/cascadia/worker@sha256:...",
    jobs=[
        ContainerInput("seed-000", args=("simulate", "--seed", "0")),
        ContainerInput("seed-001", args=("simulate", "--seed", "1")),
    ],
    resources=Resources(cpu=2, memory_gib=2, disk_gib=1),
    outputs=("/outputs",),
    timeout_seconds=3600,
    experiment_id="example-v1",
)
```

`map` returns in input order even when execution finishes out of order.
`submit_map` returns a reconnectable handle with `status`, `wait`, `cancel`, and
`results`. A request ID plus item key is idempotent: a matching specification
reattaches, while a different specification is rejected.

Large maps set `scheduler_backpressure=True`. The client atomically persists
the complete ordered logical map before its first submission, derives a safe
outstanding-job window by packing each item's declared resources against all
connected workers' advertised maximum capacities, and releases the next item
when a submitted job becomes terminal. The durable state distinguishes planned
items from admitted Bacalhau job IDs and survives a submit-before-state-write
failure by recovering the existing request/item/specification labels. This is
aggregate flow control only: the client never selects a node or creates host
waves, and Bacalhau remains responsible for placement, admission of each
released job, retry, and rescheduling.

The API rejects mutable image tags, topology fields, secret-looking environment
keys, duplicate item keys, malformed resource requests, and non-content-addressed
inputs. Every job receives separate queue, execution, publication, and total
timeouts.

The request model retains Cascadia's three-attempt application contract in job
metadata. Bacalhau v1.9 does not expose a per-job execution-attempt setting.
`Orchestrator.EvaluationBroker.MaxRetryCount=1000` protects broker deliveries;
it is not an execution-attempt count and does not expand the finite internal
over-subscription queue.

## Image and output contract

Only john1 builds images. Use `tools/cluster_build_push.py`; it builds linux/arm64,
pushes once, resolves the registry digest, and writes a source-bound publication
receipt. Before building, the tool hashes every tracked and nonignored untracked
John1 workspace file, embeds that BLAKE3 plus the Git revision and image tag in
reserved OCI labels, verifies those labels locally, and records the identity in
the publication receipt. Jobs must use the returned `@sha256:` reference.

Research images include `/usr/local/bin/cascadia-cluster-job`. Set it as the
entrypoint and pass the real command as arguments. It creates
`/outputs/manifest.json`, checksums every returned file, and prevents deterministic
application failures from being retried on every node. Transient exit codes remain
nonzero for Bacalhau recovery.

MinIO inputs use keys derived from SHA-256. Managed result objects use the immutable
`executions/<job-id>/<execution-id>.tar.gz` layout. The importer rejects traversal,
links, undeclared files, size differences, and checksum differences before an
atomic canonical import. Duplicate successful attempts remain noncanonical evidence;
the earliest valid execution is accepted.

## Operations

```bash
make cluster-fabric-install
make cluster-fabric-health
make cluster-fabric-test
make cluster-fabric-start-storage
```

The first command verifies and installs the pinned Bacalhau binary on all four
nodes. The health record requires exactly john1-john4, v1.9.0, Docker support,
and connected membership. The dashboard at `/cluster` shows the same authoritative
scheduler state beside the four-node fleet and the separate john1 MLX indicator.

Storage startup is intentionally independent from scheduler startup so native MLX
training is never disturbed by a Colima image pull. After storage is live, create
both buckets through `ObjectStoreClient.ensure_bucket` before submitting jobs.

The R2-MAP smoke and fixed-250 comparison use
`tools/r2_map_bacalhau_gate.py`. It submits one independent `pair-NNNN` item per
registered pair, reconnects by request ID, validates/imports every result
manifest, and performs the final reduction in another container. Its dashboard
watcher reads John1's imported receipts and Bacalhau request state; it never
polls worker files over SSH or binds a scientific pair to a physical node.
Cross-architecture pairs reserve 2 CPUs, 4 GiB memory, and 4 GiB disk. Against
the current 12 GiB john1 and 15 GiB john2-john4 memory allocation, the managed
admission window is derived from live scheduler capacity, with actual placement
still chosen entirely by Bacalhau.

The live compute configuration advertises 9 CPUs, 12 GiB memory, and 80 GiB
transient disk on john1, plus 10 CPUs, 15 GiB memory, and 80 GiB transient disk
on each of john2, john3, and john4. That gives the scheduler 39 CPUs, 57 GiB of
advertised memory, and 320 GiB of advertised disk. Job requests declare measured
per-item resources; Bacalhau decides packing and placement. Bacalhau disk is
scratch space only: write durable outputs under `/outputs` for S3Managed
publication, use `$CASCADIA_SCRATCH_ROOT` for temporary computation, and assume
all worker-local files vanish after the execution finishes.

## Failure behavior

- A lost caller does not cancel work; reconnect with its request ID.
- A lost worker is removed from placement and Bacalhau reschedules its execution.
- Deterministic nonretryable exits become validated failure manifests and do not
  churn across workers.
- Transient runtime exits retain their nonzero status for bounded Bacalhau retry.
- Object-store or checksum failure never imports a success.
- An oversized resource request remains visibly queued/unschedulable; resources are
  never silently reduced.
- Cancellation stops only nonterminal children and preserves terminal results.

## Version provenance

- Bacalhau: v1.9.0, commit `47cdd63c3bfe6e5e7236896151122f3c52d7c0aa`,
  Darwin arm64 SHA-256
  `adb62f07b9e0ef2122f11714ba9bc233c8a4e36d61b4044603c7dbea638bd7c7`.
- Registry: `registry@sha256:85347ed2ecde64161c7a4788a4d7d3dcc9d6f86f7be95834022e3c6a423a945a`.
- MinIO: `quay.io/minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e`.

Primary references: [Bacalhau](https://github.com/bacalhau-project/bacalhau),
[Distribution registry deployment](https://distribution.github.io/distribution/about/deploying/),
and [MinIO container deployment](https://min.io/docs/minio/container/index.html).
