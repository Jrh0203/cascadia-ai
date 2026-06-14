# Local Cluster Dashboard

The Cascadia Compute dashboard is the local control surface for the three-node
Apple Silicon research fleet:

| Node | Address | Role |
|---|---|---|
| `john1` | `100.110.109.6` | Coordinator and research workstation |
| `john2` | `100.100.43.38` | Simulation worker |
| `john3` | `100.71.97.55` | Simulation worker |

The john1 production service is available at
`http://127.0.0.1:5187/cluster`. From another machine on the tailnet, use
`http://100.110.109.6:5187/cluster`. The view refreshes every five seconds and
preserves the last successful sample when a later collection fails.

## Requirements

The API process runs on john1. Its SSH configuration must provide passwordless
aliases named `john2` and `john3`. Each worker must allow SSH over Tailscale.
The dashboard does not require an agent, Docker, or additional package on
either worker. The john1 API process owns collection and history retention.

The API invokes only a fixed metrics script. The HTTP request cannot select a
host or provide a shell command.

## Telemetry

Each sample reports:

- reachability and SSH probe duration
- normalized CPU utilization and 1/5/15-minute load average
- pressure-aware macOS memory use
- Data volume capacity
- uptime, sleep policy, automatic restart, and power source
- repository, release binary, MLX runtime, branch, revision, and dirty count
- active `cascadia-v2`, `cascadia-cli`, and `cascadia-mlx` processes

macOS `ps` CPU values are summed across processes and divided by the number of
logical cores, producing a node utilization percentage from 0 to 100.

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

The dashboard renders separate CPU and memory plots with one line per node.
The `1D` and `7D` segmented controls change the range without affecting the
live five-second node cards. History begins when the updated API first starts;
no synthetic measurements are backfilled.

The history path can be changed for an isolated run:

```bash
cargo run -p cascadia-api -- \
  --api-only \
  --cluster-history-path /tmp/cascadia-telemetry.jsonl
```

## Production Service

The checked-in launch agent serves the production frontend and API from one
release process. It keeps port 5187 available independently of a terminal and
restarts the service after an unexpected exit:

```bash
npm --prefix apps/web run build
cargo build --release -p cascadia-api
launchctl bootstrap "gui/$(id -u)" \
  tools/com.johnherrick.cascadia.dashboard.plist
```

After rebuilding, restart the loaded service with:

```bash
launchctl kickstart -k \
  "gui/$(id -u)/com.johnherrick.cascadia.dashboard"
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
cargo clippy -p cascadia-api --all-targets -- -D warnings
npm --prefix apps/web test
npm --prefix apps/web run lint
npm --prefix apps/web run build
npm --prefix apps/web run test:e2e
```

The Playwright flow exercises the real Rust endpoint and writes the dashboard
evidence images to `docs/v2/reports/web-cluster-dashboard.png` and
`docs/v2/reports/web-cluster-dashboard-mobile.png`.
