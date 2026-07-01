# Orchestrator Rewrite: Bacalhau Container Execution Fabric

**Status:** Scheduler-owned work-item cutover and 9/10/10 live deployment complete; post-training storage/image canary pending  
**Owner:** Cascadia research infrastructure  
**Decision date:** 2026-06-19  
**Target nodes:** `john1`, `john2`, `john3`  

Implementation record, 2026-06-19:

- legacy host queue frozen with a checksummed read-only cutover manifest;
- Bacalhau v1.9.0 orchestrator plus john1-john3 compute membership live;
- live health verifies 9/10/10 allocatable CPUs, 29 total, with Docker on all
  three compute nodes;
- topology-free typed Python client, durable reconnect, strict result import,
  registry/MinIO service definitions, launch supervision, and dashboard landed;
- R2-MAP focal and longitudinal schemas migrated to v4 scheduler-managed
  pair/game work items, one scientific unit per Bacalhau job;
- blinded smoke/fixed-250 Bacalhau controller implemented and unit tested;
- registry/MinIO startup, image publication, and destructive live canaries are
  deliberately waiting for the protected native john1 MLX run to release
  resources. This is a phase barrier, not an infrastructure blocker.

The compute fabric advertises 9 CPUs on john1 and 10 CPUs each on john2 and
john3, for 29 allocatable CPUs. Public callers never divide work by host,
parity, or a hand-authored partition; they submit independent scientific work
items and let Bacalhau pack them against live capacity.

## 1. Outcome

Replace the repository's host-aware SSH queue and artifact-transfer machinery with a topology-independent container execution API backed by Bacalhau.

The caller should be able to submit one Docker image plus a collection of independent inputs, wait for the work to finish, and receive an ordered collection of results without choosing machines, copying images, managing leases, or recovering failed workers.

```python
results = cluster.map(
    image="registry.cascadia/worker@sha256:...",
    jobs=[
        ContainerInput(key="shard-0", args=["simulate", "--seed-start", "0"]),
        ContainerInput(key="shard-1", args=["simulate", "--seed-start", "10000"]),
        ContainerInput(key="shard-2", args=["simulate", "--seed-start", "20000"]),
    ],
    resources=Resources(cpu=10, memory_gib=24, disk_gib=10),
    outputs=["/outputs"],
    timeout_seconds=3600,
)
```

Bacalhau owns node selection, queueing, admission, execution attempts, retries, rescheduling, cancellation, and job state. Cascadia owns the typed caller API, scientific metadata, deterministic aggregation, artifact validation, and experiment records.

## 2. Scope

### In scope

- CPU-bound Docker workloads on `john1`, `john2`, and `john3`.
- Simulation, self-play generation, benchmarks, evaluation, data transformation, and other independent batch work.
- One-to-many job submission and ordered result aggregation.
- Automatic scheduling around available CPU and memory.
- Worker and container failure recovery.
- Immutable image distribution from a registry on `john1`.
- Durable result publication to object storage on `john1`.
- Bacalhau job status in the existing cluster dashboard.
- Migration of new research campaigns away from SSH dispatch.

### Out of scope

- MLX training. MLX remains a dedicated flow run directly on `john1`.
- Distributed training or cross-node model synchronization.
- Arbitrary untrusted or multi-tenant workloads.
- Scheduling on `john4`.
- Replacing the scientific experiment ledger with the scheduler database.

## 3. Trust and security decision

The `john1`–`john3` cluster is a private, trusted topology. Its operators, source tree, images, Docker daemons, and submitted workloads are trusted. Container sandbox hardening is therefore not a requirement or a cutover gate.

By owner direction on 2026-06-19, the implementation does **not** need to enforce:

- a read-only root filesystem;
- an image-defined non-root user;
- dropping all Linux capabilities;
- `no-new-privileges`;
- PID limits;
- bounded `tmpfs` mounts;
- read-only input mounts;
- exactly one writable output mount;
- network-disabled-by-default execution;
- prohibitions on Docker socket, home-directory, or source-tree mounts.

Standard Bacalhau Docker execution is acceptable. Jobs may use normal writable container filesystems, run as the image's configured user or root, use ordinary Docker capabilities and security defaults, access the network, and receive the mounts required by the workload. Bacalhau workers may access their local Colima Docker daemons.

This decision removes the need for a custom Bacalhau executor, a Bacalhau fork, an upstream hardening patch, or a separate container-policy admission layer.

The following controls remain because they protect reproducibility and operational reliability, not because the nodes are considered hostile:

- images referenced by immutable digest;
- explicit CPU, memory, disk, and timeout requests;
- unique execution output paths;
- checksums and manifests for returned artifacts;
- idempotent submission and result import;
- secrets kept out of images, command arguments, and logs;
- Bacalhau control and storage endpoints restricted to the private cluster network.

If the topology later admits untrusted workloads or users, isolation must be reconsidered as a separate project before that trust boundary changes.

## 4. Target architecture

```text
                          topology-free API
 Research code  ------------------------------------+
                                                     |
                                                     v
                    +----------------------------------------+
                    | cascadia_cluster client                |
                    | map / submit_map / status / cancel     |
                    | idempotency / ordering / aggregation   |
                    +--------------------+-------------------+
                                         |
                                         | Bacalhau REST API
                                         v
  john1         +------------------------------------------------+
                | Bacalhau orchestrator + Web UI                 |
                +----------+---------------------+---------------+
                           |                     |
             +-------------+-------+     +-------+----------------+
             | OCI registry        |     | MinIO result store     |
             | immutable images    |     | inputs / outputs       |
             +---------------------+     +------------------------+
                           |
              scheduling and execution over private network
                           |
           +---------------+---------------+---------------+
           v                               v               v
  +------------------+            +------------------+  +------------------+
  | john1 worker     |            | john2 worker     |  | john3 worker     |
  | Colima + Docker  |            | Colima + Docker  |  | Colima + Docker  |
  +------------------+            +------------------+  +------------------+
```

`john1` is the source of truth for code, images, job metadata, and accepted artifacts. `john2` and `john3` are fungible execution capacity. Callers never address a worker by name.

## 5. Ownership boundaries

| Concern | Owner |
|---|---|
| Node selection, queueing, admission, retries, rescheduling | Bacalhau |
| Docker lifecycle on each worker | Bacalhau Docker executor + local Colima |
| Image build and publication | `john1` CI/build command + private OCI registry |
| Typed submission, idempotency, reconnect, result ordering | `cascadia_cluster` client |
| Input and output byte transport | MinIO through Bacalhau S3 input/publisher support |
| Artifact validation and canonical import | Cascadia result importer |
| Experiment hypotheses, configurations, metrics, and conclusions | Existing scientific ledger |
| Fleet and job observability | Existing cluster dashboard + Bacalhau API |
| MLX training | Dedicated `john1` flow outside Bacalhau |

Do not reproduce Bacalhau's live job state in the legacy manifest queue. The scheduler is authoritative for execution state; the experiment ledger stores durable scientific provenance.

## 6. Migration phases

### Phase 0: Freeze and preserve the legacy queue

At plan creation, the legacy queue reports no ready or running tasks. Use this as the migration boundary.

1. Snapshot the current queue manifest, task records, attempts, and artifact references.
2. Preserve existing queue data as read-only historical evidence.
3. Stop creating new host-specific container tasks.
4. Keep the old dispatcher available only for rollback during the bounded cutover window.
5. Do not rewrite or discard completed research records.

### Phase 1: Install the Bacalhau fabric

1. Pin Bacalhau to a tested release, initially `v1.9.0`, on all three nodes.
2. Run the Bacalhau orchestrator on `john1`.
3. Run native Bacalhau compute workers on `john1`, `john2`, and `john3`.
4. Point each worker's Docker executor at its local Colima Docker daemon.
5. Install each service under `launchd` with restart-on-failure and persistent state directories.
6. Bind API, NATS, and Web UI traffic to the private Tailscale network.
7. Advertise conservative allocatable CPU, memory, and disk values so macOS and Colima retain operating headroom.
8. Apply internal worker labels for diagnostics only. Public submission APIs must not accept or expose node affinity.
9. Add health checks for orchestrator reachability, worker membership, Docker readiness, registry access, and object-store access.

### Phase 2: Replace image copying with an OCI registry

1. Run a private OCI registry on `john1` with persistent storage.
2. Make `john1` the only supported image build source.
3. Build each research image once, tag it for humans, push it, resolve the pushed digest, and submit jobs by digest.
4. Configure all workers to pull from the private registry and benefit from local layer caching.
5. Record image digest, source commit, build timestamp, and build configuration in each experiment record.
6. Remove `docker save`, `rsync`, and `docker load` from the new execution path.

### Phase 3: Establish durable input and result transport

1. Run MinIO on `john1` with persistent buckets for `cascadia-inputs` and `cascadia-results`.
2. Stage large inputs by content hash and reuse them across jobs.
3. Use Bacalhau's managed S3 publisher for job outputs.
4. Publish every execution into a unique prefix containing request ID, item ID, and execution ID.
5. Require an output manifest containing paths, byte sizes, SHA-256 checksums, command metadata, and application result metadata.
6. Download or stream successful outputs to `john1`, validate the manifest and checksums, then atomically import them into the canonical artifact tree.
7. Keep rejected, duplicate, and failed-attempt outputs outside the canonical artifact namespace for diagnosis and retention cleanup.

### Phase 4: Build the topology-free client

Add a small maintained Python package:

```text
python/cascadia_cluster/
  __init__.py
  client.py
  models.py
  results.py
  errors.py
  bacalhau_api.py
```

Use Bacalhau's REST API directly behind `bacalhau_api.py`. This keeps the public Cascadia API stable if Bacalhau changes its command-line output or SDK surface.

Required synchronous API:

```python
cluster.map(image, jobs, resources, outputs, timeout_seconds) -> list[JobResult]
```

Required asynchronous API:

```python
handle = cluster.submit_map(...)
handle.status()
handle.wait()
handle.cancel()
handle.results()
```

Each `ContainerInput` becomes an independent Bacalhau job. Bacalhau's replica count is not used to represent different shards because replicas have identical specifications and inputs.

The public API must reject topology-bearing fields such as `host`, `node`, `compatible_hosts`, remote roots, and SSH commands.

### Phase 5: Define stable contracts

Core request models:

- `ContainerSpec`: immutable image digest, entrypoint override, environment, working directory, mounts, and output paths.
- `ContainerInput`: stable item key, arguments, environment overrides, and content-addressed input references.
- `Resources`: CPU, memory, ephemeral disk, optional GPU metadata reserved for future use, and execution timeout.
- `RetryPolicy`: maximum attempts, retryable exit conditions, and backoff bounds.

Core result models:

- `JobResult`: item key, request ID, Bacalhau job ID, accepted execution ID, status, exit code, timestamps, logs reference, artifact manifest, and application metadata.
- `MapResult`: results in the same order as submitted inputs plus aggregate timing and failure counts.
- `MapError`: structured failures with all successful partial results preserved.

The default blocking call returns only after every item reaches a terminal state. Successful results retain input order regardless of completion order. Partial failure is explicit and machine-readable rather than silently dropping results.

### Phase 6: Make submission and collection idempotent

Derive a stable specification hash from:

- image digest;
- item key;
- command and arguments;
- non-secret environment;
- input object checksums;
- resource and timeout declarations;
- requested output paths;
- application protocol version.

Generate a request ID for each logical `map` invocation and attach `request_id`, `item_id`, `spec_sha256`, and `experiment_id` labels to each Bacalhau job.

Bacalhau also treats its human-readable job `Name` as an update key. Derive
that name from a collision-resistant hash of the complete request ID plus the
stable item index/key; never truncate the request ID into a suffix. Reducer
request IDs additionally bind the assembled input archive hash so an obsolete
or failed reducer can never be reused for different campaign bytes.

On client retry or restart:

1. Query existing jobs by labels.
2. Reattach to matching nonterminal jobs.
3. Reuse a validated success when its specification hash matches.
4. Submit only missing items.
5. Reject conflicting reuse of an item key with a different specification hash.

For a logical map larger than Bacalhau v1.9's finite internal
over-subscription queue, persist every planned item first and apply durable
scheduler-capacity backpressure. Derive the maximum outstanding count by
packing the declared per-item CPU, memory, disk, and GPU requirements against
connected workers' advertised maximum capacities, summed across the fabric.
Submit items in stable input order as terminal jobs free slots. This admission
window is aggregate flow control only: it must not select a node, create host
batches, or make physical placement part of request identity. Reconnect must
resume partial admission and recover a submit-before-state-write interruption
through the existing request/item/spec labels without duplicating a job.

Execution semantics are at-least-once. Application jobs must therefore write to execution-specific paths. The importer accepts exactly one valid successful execution per item and atomically records that choice.

Define separate queue, execution, publication, and total request timeouts. A lost caller must not cancel running work by default; it must be able to reconnect later using the request ID.

### Phase 7: Integrate research workflows

1. Replace campaign host shards with `ContainerInput` records.
2. Translate each existing independent simulation, self-play, or benchmark shard into `cluster.map` or `submit_map`.
3. Remove `compatible_hosts`, SSH roots, worker-specific commands, manual image fanout, queue leases, and artifact `rsync` from migrated workflows.
4. Import Bacalhau request IDs, job IDs, image digests, input hashes, output checksums, and timing into the scientific experiment ledger.
5. Pool deterministic tabular outputs in item order; make domain-specific reducers explicit and independently testable.
6. Keep training checkpoints and MLX state outside this scheduler and link them through artifact IDs only.

### Phase 8: Integrate observability

Retain the compact fleet view for `john1` through `john4`. The scheduler manages only `john1` through `john3`, but `john4` remains visible as fleet capacity occupied elsewhere.

Add a concise Bacalhau section showing:

- queued, running, successful, retrying, and failed job counts;
- per-node allocated versus available CPU and memory;
- current jobs by request and experiment;
- attempt count and most recent failure reason;
- queue time and execution duration;
- links to Bacalhau job detail and logs;
- registry and object-store health.

MLX training remains a separate, unambiguous status on the `john1` card. Do not infer MLX state from Bacalhau activity.

## 7. Failure behavior

| Failure | Expected behavior |
|---|---|
| Container exits nonzero | Record attempt; retry only according to policy; surface final stderr and exit code |
| Worker disappears | Bacalhau marks execution lost and reschedules on another healthy worker |
| Colima stops on a worker | Worker becomes unavailable; other workers continue; health alarm identifies Docker failure |
| Caller exits or disconnects | Jobs continue; caller reconnects by request ID |
| Orchestrator restarts | Persistent Bacalhau state restores job tracking; clients retry API calls |
| Registry temporarily fails | Pull is retried with bounded backoff; already cached images remain usable |
| Object store is unavailable | Execution or publication remains incomplete and retries; no success is imported |
| Output checksum fails | Reject the execution output and report artifact corruption |
| Duplicate execution succeeds | Atomically accept one valid result and retain the duplicate only for diagnostics |
| Job cannot fit any worker | Keep queued with a clear unschedulable reason; do not silently reduce resources |
| Request is cancelled | Cancel outstanding Bacalhau jobs, preserve terminal results, and return structured cancellation state |

Retries must be bounded. Deterministic application errors should not churn through the cluster, while lost workers and transient infrastructure failures should be retried automatically.

## 8. Verification strategy

### Unit tests

- request validation and rejection of topology fields;
- stable specification hashing;
- REST request and response translation;
- status normalization;
- ordered aggregation under out-of-order completion;
- partial failure representation;
- retry classification;
- duplicate-success arbitration;
- output manifest and checksum validation;
- atomic artifact import;
- reconnect and idempotent resubmission.

### Single-node integration tests

- submit a known image and receive stdout plus an artifact;
- pass staged input through MinIO;
- publish multiple output files;
- verify timeout, cancellation, nonzero exit, and retry behavior;
- restart the caller and reattach;
- restart the orchestrator and recover state;
- verify an image is addressed and recorded by digest.

### Three-node acceptance tests

- submit at least 100 independently seeded jobs and observe scheduling across every healthy worker;
- submit enough work to exercise queueing and resource admission;
- kill a running container and verify bounded recovery;
- stop a worker during execution and verify rescheduling without caller intervention;
- stop Colima on one worker and verify the remaining fleet drains the queue;
- disconnect and restart the caller while work is running;
- interrupt the object store and verify that incomplete publication is never reported as success;
- deliberately corrupt an output and verify checksum rejection;
- submit an oversized job and verify an explicit unschedulable state;
- cancel a map request and verify all nonterminal children are cancelled;
- run 1,000 small jobs to expose scheduler, connection, and artifact-store bottlenecks.

### Scientific parity test

Run a deterministic representative campaign through both the legacy executor and Bacalhau. Given the same image digest, arguments, inputs, and seeds, compare output manifests and application results byte-for-byte where formats are deterministic. Investigate every difference before cutover.

## 9. Cutover

1. Complete unit and single-node integration tests.
2. Run the three-node acceptance suite.
3. Run scientific parity canaries against the frozen legacy executor.
4. Migrate one real, bounded research campaign and inspect its scheduler and artifact records.
5. Change the default cluster backend for all new container campaigns to Bacalhau.
6. Disable new submissions to the legacy manifest queue. Never run both schedulers for the same logical item.
7. Remove legacy dispatch and image/artifact-copy commands from normal Make targets and documentation.
8. Retain the old queue reader and historical files for auditability during a defined retention period.
9. After the retention window and successful production use, remove dead SSH execution paths while preserving historical data readers.

## 10. Rollback

Rollback must not destroy scheduler or artifact state.

1. Stop new Bacalhau submissions.
2. Allow safe running jobs to finish or explicitly cancel them.
3. Preserve Bacalhau state, MinIO objects, job receipts, and imported artifacts.
4. During the bounded transition window only, switch the client backend to the frozen legacy executor for new request IDs.
5. Never replay an item whose accepted success already exists.
6. Correct the fault, rerun acceptance tests, and resume Bacalhau submission.

## 11. Repository changes

Expected implementation surface:

```text
infra/bacalhau/                  # pinned configuration and launchd definitions
infra/registry/                  # registry configuration and persistence docs
infra/minio/                     # buckets, credentials wiring, retention configuration
python/cascadia_cluster/         # public client and Bacalhau adapter
tests/cluster_unit/              # client and artifact contract tests
tests/cluster_integration/       # local and three-node test harness
tools/cluster_build_push.py      # build once, push, and resolve digest
tools/cluster_artifact_import.py # validate and atomically import results
docs/cluster_orchestrator.md     # operator and caller documentation
```

Update the Makefile with explicit install, start, stop, health, build-push, test, canary, and migration targets. Update the API/dashboard backend to read Bacalhau for live execution state while retaining historical legacy queue views.

Do not mechanically delete `cluster_research_queue.py`, `cluster_artifact_collect.py`, or `cluster_artifact_fanout.py` until their remaining callers have been identified and migrated. Once unused, move their historical reading responsibility into a small read-only compatibility module and remove their execution paths.

## 12. Acceptance criteria

The rewrite is complete when all of the following are true:

1. A caller can run a Docker image over a list of inputs and receive ordered results without naming a node.
2. `john1` builds an image once, publishes it once, and all workers execute the immutable digest without manual copying.
3. Bacalhau schedules work across all healthy capacity on `john1`, `john2`, and `john3`.
4. Queueing respects declared resources without oversubscribing the workers.
5. Container and worker failures produce bounded retries and automatic rescheduling.
6. Caller restart and network interruption do not lose or duplicate logical work.
7. Every accepted artifact has a request ID, item ID, job ID, execution ID, image digest, and validated checksum manifest.
8. Partial failures are returned explicitly with all successful results preserved.
9. The dashboard shows authoritative fleet and Bacalhau execution state at a glance, with MLX clearly separate.
10. New research campaigns contain no SSH dispatch, host compatibility list, manual lease, image fanout, or result `rsync` logic.
11. MLX training remains unaffected and continues through its dedicated `john1` flow.
12. The entire unit, integration, failure, scale, and scientific parity test suite passes.
13. No custom container-hardening executor, Bacalhau fork, or security-policy gate exists for this trusted topology.

## 13. Definition of done

This is not done when Bacalhau merely runs a container. It is done when the old topology-aware execution path has been replaced end to end: build once, submit without host knowledge, schedule across the cluster, survive failures, publish and validate artifacts, return ordered results, expose truthful status, preserve scientific provenance, and pass the full acceptance suite.
