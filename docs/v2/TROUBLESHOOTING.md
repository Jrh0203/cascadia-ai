# Troubleshooting

Run commands from the repository root on john1 unless a runbook explicitly
names another host.

## First Checks

```bash
make mlx-device
make format-check
make lint
make test
```

`make mlx-device` must report `Device(gpu, 0)`. The remaining commands must
finish without warnings or failures before diagnosing an experiment result.

## MLX Or Python

Use the checked-in uv environment, not an arbitrary system Python:

```bash
uv sync --all-groups
uv run python --version
uv run cascadia-mlx-device
```

If a standalone `python`, `mlx`, or `tailscale` executable fails with a dyld
framework error, verify which binary is being invoked with `command -v`. Do
not copy missing frameworks or alter system library paths; use the uv-managed
Python environment or reinstall the affected application normally.

## Cluster SSH

Check each worker without opening an interactive shell:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 john2 'hostname'
ssh -o BatchMode=yes -o ConnectTimeout=8 john3 'hostname'
```

An SSH timeout usually means the Mac is asleep, powered off, or absent from
Tailscale. Verify power and Tailscale reachability before changing keys.
Authentication failures require checking the worker user's
`~/.ssh/authorized_keys` permissions and the matching local SSH alias.

Never resume a distributed shard until its source revision, executable hash,
split, and index range match the owning manifest.

## Resumable Data And Training

Dataset and checkpoint writers fail closed on source, schema, configuration,
or checksum drift. Read the reported mismatch rather than deleting the
artifact. Resume only with the original command and the explicit `--resume`
path.

For ADR 0078/0079:

```bash
cat artifacts/logs/adr0078-cluster-supervisor-state.json
tail -n 100 artifacts/logs/adr0078-cluster-supervisor.log
launchctl print gui/$(id -u)/com.johnherrick.cascadia.adr0078-supervisor
```

The supervisor may reclaim a lock only when its recorded PID is no longer
alive. Do not remove a live lock or manually create sealed-test data.

ADR 0078/0079 SSH uses the `john2` Tailscale alias first. If that route times
out, the transport retries `john2@192.168.1.238` with
`StrictHostKeyChecking=yes` and `HostKeyAlias=100.100.43.38`; the LAN route is
therefore accepted only when it presents john2's already pinned host key.
The same fallback applies to rsync. A remote command exit other than SSH's 255
transport failure is never retried through another endpoint.

Validation is owned exclusively by john2. If an unintended john1 collector
creates the same dataset, stop that process before restarting the supervisor.
The handoff can archive and replace only a byte-identical strict prefix with
the same immutable contract. Any other collision remains a hard failure.
Archived collision evidence lives under `artifacts/datasets/invalidated/`.

## Web And Dashboard

Development:

```bash
make web-dev
```

Production dashboard:

```bash
launchctl print gui/$(id -u)/com.johnherrick.cascadia.dashboard
curl -fsS http://127.0.0.1:5187/api/v1/cluster
curl -fsS 'http://127.0.0.1:5187/api/v1/cluster/history?range=1d'
```

If port 5187 is occupied, inspect the owner with
`lsof -nP -iTCP:5187 -sTCP:LISTEN`. Do not run a second dashboard server over
the launchd service. Rebuild and restart using
[`CLUSTER_DASHBOARD.md`](CLUSTER_DASHBOARD.md).

For browser failures, run `make web-test` and inspect
`artifacts/web-test-results/`. Confirm the Rust API is reachable before
debugging frontend selectors.

## Generated Artifacts

All substantive datasets, checkpoints, reports, and telemetry belong under
`artifacts/` with manifests or documented retention rules. Do not rename
statistical artifacts to imply promotion. Preserve failed or partial evidence
when an ADR requires it; remove only regenerated caches and build outputs.

## Reproducibility Escalation

When a result cannot be reproduced:

1. Compare source and executable hashes.
2. Compare typed configuration and seed/index ranges.
3. Validate every input manifest and shard.
4. Confirm hardware, OS, Python, Rust, and MLX versions.
5. Re-run the smallest registered smoke without changing the protocol.
6. Record a new ADR before changing data, thresholds, seeds, or architecture.
