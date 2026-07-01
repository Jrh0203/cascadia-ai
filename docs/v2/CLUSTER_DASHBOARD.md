# Local Cluster Dashboard

## Bacalhau execution fabric

The top-level dashboard keeps three truths separate:

1. the compact john1-john4 fleet cards show host CPU, memory, disk, and native MLX;
2. the Bacalhau strip shows john1-john3 placement capacity, job state, registry,
   and artifact-store health;
3. the R2-MAP panel shows scientific campaign progress and model evidence.

Scheduler state comes directly from the Bacalhau REST API. It is never reconstructed
from the frozen legacy queue. MLX state is never inferred from scheduler activity.

The Cascadia Compute dashboard is the local control surface for the four-node
Apple Silicon research fleet:

| Node | Address | Role |
|---|---|---|
| `john1` | `100.110.109.6` | Coordinator and research workstation |
| `john2` | `100.100.43.38` | Simulation worker |
| `john3` | `100.71.97.55` | Simulation worker |
| `john4` | `100.118.7.103` | Simulation worker |

All four Macs remain visible in fleet telemetry. A research campaign may reserve
only a subset of them; the current R2-MAP status contract assigns John1–3 and
shows John4 as available or working outside that campaign.

The john1 production service is available at
`http://127.0.0.1:5187/cluster`. From another machine on the tailnet, use
`http://100.110.109.6:5187/cluster`. The view refreshes every five seconds and
preserves the last successful sample when a later collection fails.

The page is a phase-aware command deck rather than a vertically stacked
telemetry report. Its default view contains the current R2-MAP phase,
phase-specific progress, the next transition, alerts, and a permanent compact
**Fleet status** strip for John1–4 as the first content beneath the application
bar. Every node shows current CPU, memory, and
disk utilization as proportional circular rings with centered percentages and
green/amber/red pressure coloring, plus an explicit MLX-training yes/no signal. CPU and memory
history are also permanently visible at the top level with `1D` and `7D`
ranges. Deeper campaign evidence is mutually exclusive behind four tabs:

- **Training** — loss history, verified checkpoint, and John1 memory state;
- **Benchmark** — pair progress, score tails, paired delta, and score anatomy;
- **Fleet** — full node resources and active process commands;
- **Research** — the experiment decision ledger and scheduler queue.

Raw host status strings, model identities, and opponent-pool metadata are
collapsed under **Campaign metadata**. The command deck combines the three-host
R2-MAP status with four-host fleet telemetry without claiming John4 is part of
the campaign.

## Requirements

The API process runs on john1. Its read-only fleet telemetry uses fixed
passwordless Tailscale aliases for `john2`, `john3`, and `john4`; browser input
cannot alter a host or command. Workload scheduling and job monitoring do not
use this SSH path and come directly from Bacalhau.
The dashboard does not require an agent, Docker, or additional package on
the workers. The john1 API process owns collection and history retention.

The dashboard may still read the frozen legacy queue at
`artifacts/cluster/research-queue-v1.json` as historical evidence. It is not
execution state and cannot be mutated through the active scheduler. Current
queue mechanics are documented in [CLUSTER_SCHEDULER.md](CLUSTER_SCHEDULER.md).

The durable experiment ledger lives at
`artifacts/cluster/research-experiments-v1.json`. It is intentionally separate
from the scheduler: queue tasks describe execution state, while experiment
records describe hypotheses, preregistered gates, observations, scientific
verdicts, and durable artifact paths.

The API invokes only a fixed metrics script. The HTTP request cannot select a
host or provide a shell command.

An SSH telemetry probe can wake a sleeping Mac only into DarkWake. Bacalhau's
native compute service and Colima lifecycle own workload availability; the
dashboard probe is observational only.

## Telemetry

Each sample reports:

- reachability and SSH probe duration
- normalized CPU utilization and 1/5/15-minute load average
- pressure-aware macOS memory use
- Data volume capacity
- uptime, sleep policy, automatic restart, and power source
- repository, release binary, MLX runtime, branch, revision, and dirty count
- active `cascadia-v2`, `cascadia-cli`, and `cascadia-mlx` processes

Each probe captures one macOS `ps` snapshot. CPU values for every process and
the active-workload table are derived from that same snapshot. Process CPU is
summed and divided by logical-core count to produce node utilization from 0 to
100. Fleet CPU is the sum of used logical cores divided by total online
logical cores, so heterogeneous nodes are weighted by capacity rather than
averaged as peers.

## Utilization History

The API samples the fleet every 30 seconds even when no dashboard tab is open.
Successful snapshots are appended to
`artifacts/cluster/telemetry-v1.jsonl`. The journal:

- retains seven rolling days;
- rejects duplicate samples closer than 25 seconds;
- recovers an incomplete final JSONL record after an interrupted write;
- rejects corruption in any completed record;
- periodically compacts expired records through an atomic replacement.

`GET /api/v1/cluster/history?range=1d` returns the 24-hour view and
`range=7d` returns the seven-day view. The server aggregates each node to at
most 480 chart points, preserves offline gaps, and reports raw sample count,
capture interval, reachability, mean, and peak CPU and memory utilization.

The dashboard always renders separate top-level CPU and memory plots with one line per node.
The `1D` and `7D` segmented controls change the range without affecting the
live five-second node cards. History begins when the updated API first starts;
no synthetic measurements are backfilled.

## Frozen legacy queue

`GET /api/v1/cluster/queue` returns the preserved pre-cutover campaign and task
history. It never returns claim tokens or command arguments and must not be
interpreted as live Bacalhau state.

The historical view can show:

- running, ready, blocked, and completed task counts;
- active replica count and ready critical-path count;
- each host's scheduler intent and reason;
- open tasks ordered by status, priority, critical path, and identifier; and
- the assigned host or compatible host set, workload class, resources, and
  expected duration.

Missing queue state is a valid unconfigured condition. Malformed historical
state is shown as an archive error without taking fleet telemetry, Bacalhau,
or game APIs down.

## Research Experiments

`GET /api/v1/cluster/experiments` returns the durable research decision ledger.
The dashboard polls it every five seconds and orders running work first,
followed by the newest completed decisions. Each row can be expanded to show:

- the hypothesis and formal verdict;
- success criteria with passed, failed, or pending state;
- compact headline metrics;
- research notes and participating hosts;
- scheduler task identifiers; and
- repository-relative paths for preregistrations, reports, and raw artifacts.

Completed treatments are explicitly labeled `Passed`, `Failed`,
`Inconclusive`, or `Invalid`. A failed treatment can therefore coexist with a
successful forensic experiment that explains the failure.

The ledger is schema-validated and written atomically:

```bash
.venv/bin/python tools/cluster_experiment_ledger.py validate
.venv/bin/python tools/cluster_experiment_ledger.py status
.venv/bin/python tools/cluster_experiment_ledger.py upsert \
  --spec /path/to/experiment-record.json
```

Missing ledger state is a valid unconfigured condition. Malformed state is
reported inside the experiment panel without affecting telemetry, history,
queue, or game APIs.

## R2-MAP Expert Iteration

`GET /api/v1/cluster/r2-map` reads exactly one compact, atomically replaced
serving projection. The production default is
`artifacts/cluster/r2-map-dashboard-serving-projection-v2.json` on John1. The
request never contacts a worker and never scans datasets, checkpoints, pair
receipts, or campaign directories. A missing projection is explicitly
`unconfigured`; a mirror older than its declared threshold is `stale`; and an
oversized, malformed, unsupported, or semantically inconsistent mirror is
`invalid` without affecting the rest of the API.

The version-1 mirror uses schema id
`cascadia.r2-map.dashboard-status.v1` and publishes:

- campaign id, phase, legal next transitions, and round index;
- incumbent, candidate, and frozen opponent-pool identities;
- exactly `john1`, `john2`, and `john3`, with each host's intent, generation
  and benchmark progress, seed partition, ETA, throughput, RSS, and swap delta;
- John1 MLX training step, latest verified checkpoint, examples per second,
  and at most 512 strictly ordered train/validation loss samples;
- benchmark stage, pair progress, ETA, throughput, peak RSS, swap delta, paired
  delta with 95% interval, and promotion classification; and
- focal score mean/P10/P50/P90 for the total, every wildlife and terrain
  component plus aggregates, and Pinecone earning, spend, remainder, and free
  replacement accounting.

When `training.active` is true and `hosts.john1.intent` is `train`, the command
deck becomes **MLX training on John1** and shows current/total step, latest
loss, latest verified checkpoint, ETA, examples per second, and epoch progress.
The host strip labels John1 **MLX training** and keeps John2/John3 intent visible
without repeating completed generation counters. RSS, swap delta, the loss
chart, and raw machine status remain available in the Training tab. These are
the authoritative visual signals that MLX is running; aggregate CPU idle
percentage is not, because Metal work and streamed CPU batch preparation
alternate during training.

When a completed lagged greedy baseline remains in the status mirror while a
new phase is active, the Benchmark tab labels it **Baseline complete** rather
than displaying its historical `promote` classification as though it were a
new candidate decision.

During John1 generation, its existing `hosts.john1.detail` field must expose
the runtime-stage state without adding a second control channel. Host detail is
limited to 512 UTF-8 bytes. The canonical forms are:

```text
runtime-stage:verified run=<id> sha256=<64hex> bytes=<n> cleanup=pending
runtime-stage:cleanup-verified run=<id> receipt_sha256=<64hex>
runtime-stage:blocked run=<id> reason=<bounded-reason>
```

`verified` is published only after path, owner, modes, <=64-MiB combined size,
manifest <=64 KiB, arm64 architecture, hashes, source/build binding, and code
signature all pass. The generation phase cannot complete while cleanup is
pending or blocked. The detail is status only; the executable, manifest, and
cleanup receipt remain outside the <=64-KiB serving projection payload except
for their identities.

The canonical status is owned by John1 at
`/Users/johnherrick/cascadia-bench/r2-map-v1/control/dashboard-status.json`.
The reader limits the serving projection to 64 KiB and accepts status staleness
thresholds from 5 to 3,600 seconds. Model digests, when present, are
64-character BLAKE3 hex. It rejects progress beyond declared totals, invalid
distributions, duplicate models, non-increasing loss steps, non-finite metrics,
unexpected hosts such as `john4`, and timestamps more than 60 seconds in the
future. The campaign controller is the only logical writer. It writes a sibling
temporary file, flushes the file and parent directory, and atomically renames it
over the canonical status on the same internal APFS volume.

The canonical publisher runs on John1 and consumes only explicit compact inputs
beneath the owner-private primary root. Its reviewed run specification binds an
immutable source identity, every input path, the canonical output, a unique
run-log directory, and a bounded timeout. The publisher may mutate only the
primary root's `control/` subtree. It rejects paths, executables, environment
values, or outputs outside that boundary. Every optional path is named
directly; the publisher never enumerates a bulk directory. Omit inputs that do
not yet exist. For example, `contracts-ready` publishes null
model/training/benchmark values instead of inventing progress.

### Local serving projection

The production API never scans the primary campaign tree during an HTTP
request. A reviewed local projection operation reads exactly the canonical
status path above, verifies schema, size, BLAKE3, timestamp, owner, mode, and
internal-volume identity, then atomically writes the disposable projection
`artifacts/cluster/r2-map-dashboard-serving-projection-v2.json`.

The projection:

- is atomically generated on John1 only after a verified local canonical read;
- is limited to 64 KiB and installed read-only;
- contains only the exact canonical JSON payload, canonical host/path, BLAKE3,
  canonical update timestamp, and fetch timestamp—never model, checkpoint,
  dataset, benchmark receipt, or log payloads;
- is cryptographically and temporally verified by the Rust reader before the
  status is accepted; and
- is non-authoritative and can be deleted and recreated without scientific
  loss.

A missing projection is `unconfigured`. Hash or timestamp drift is `invalid`.
Age is still computed from the canonical update timestamp, so an unchanged
projection becomes `stale` normally.

During an active campaign, run the canonical publisher and read-only local
projector every ten seconds on John1. Neither is another scheduler and neither
launches games or training. If the canonical file is absent or invalid, or the
projector stops, the last valid projection remains byte-identical and naturally
becomes `stale` within 30 seconds; the projector never fabricates freshness.
The existing fixed-John2 SSH fetch path and launch agent are obsolete for
R2-MAP and must not run until replaced by the reviewed local projector. No
dashboard process reads John2 cold storage or writes `/Volumes/John_1`.

The history path can be changed for an isolated run:

```bash
cargo run -p cascadia-api -- \
  --api-only \
  --cluster-history-path /tmp/cascadia-telemetry.jsonl \
  --cluster-queue-path /tmp/cascadia-research-queue.json \
  --cluster-experiments-path /tmp/cascadia-research-experiments.json \
  --r2-map-status-path /tmp/cascadia-r2-map-dashboard-status.json
```

## Production Service

The checked-in launch agent serves the production frontend and API from one
release process. It keeps port 5187 available independently of a terminal and
restarts the service after an unexpected exit:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python .venv/bin/python \
  tools/r2_map_remote_storage.py run \
  --specification /path/to/controller-issued-dashboard-build.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python .venv/bin/python \
  tools/r2_map_remote_storage.py deploy-dashboard-api \
  --relative bundles/dashboard-api-VERIFIED_BUILD_ID/cascadia-api \
  --expected-sha256 VERIFIED_BUILD_SHA256
launchctl bootstrap "gui/$(id -u)" \
  tools/com.johnherrick.cascadia.dashboard.plist
```

The build specification names the immutable source freeze and uses only the
John2 builder's sandboxed Cargo environment. Compilation and complete build
trees remain bounded, non-authoritative John2 staging. John2 returns the
immutable bundle and build evidence to John1; John1 verifies and accepts them
before the fixed-path deploy command binds the hash-verified installed API
executable to that evidence. That executable, the existing served frontend, and disposable status
projection are bounded John1 control-plane deployment assets; none is
authoritative campaign state. A frontend change follows the same
build/manifest/deploy pattern rather than `scp`, `rsync`, a generic
fetch-to-file command, or mutable tree synchronization.

After a binary-only rebuild, restart the loaded service with:

```bash
launchctl kickstart -k \
  "gui/$(id -u)/com.johnherrick.cascadia.dashboard"
```

`kickstart` reuses the already loaded launch-agent arguments. When the plist
changes (for example, the v1-to-v2 R2-MAP projection path cutover), reload the
definition instead:

```bash
launchctl bootout \
  "gui/$(id -u)/com.johnherrick.cascadia.dashboard"
launchctl bootstrap "gui/$(id -u)" \
  tools/com.johnherrick.cascadia.dashboard.plist
```

For frontend development with hot reload, stop the production agent, run
`make web-dev`, then bootstrap the agent again afterward.

## Health States

- `Ready`: reachable with no resource or power warning
- `Working`: a Cascadia job is active or CPU utilization is at least 70%
- `Attention`: memory or disk is at least 90%, load exceeds 1.5 times core
  count, system sleep is enabled, or automatic restart is disabled
- `Offline`: the local or SSH probe did not complete successfully

The cluster summary treats both `Attention` and `Offline` nodes as degraded.

## Verification

```bash
cargo test -p cascadia-api cluster --lib
cargo test -p cascadia-api cluster_queue --lib
cargo test -p cascadia-api cluster_experiments --lib
cargo test -p cascadia-api cluster_r2_map --lib
cargo clippy -p cascadia-api --all-targets -- -D warnings
.venv/bin/pytest tools/test_cluster_experiment_ledger.py
.venv/bin/pytest tools/test_r2_map_dashboard_fetch.py
npm --prefix apps/web test
npm --prefix apps/web run lint
npm --prefix apps/web run build
npm --prefix apps/web run test:e2e
```

The Playwright flow exercises the real Rust endpoint and writes the dashboard
evidence images to `docs/v2/reports/web-cluster-dashboard.png` and
`docs/v2/reports/web-cluster-dashboard-mobile.png`.
