# Bacalhau fabric configuration

This directory is the version-controlled source for the private Cascadia execution
fabric. Runtime state and credentials live outside the repository:

- john1: `/Users/johnherrick/cascadia-bench/orchestrator`
- john2/john3/john4: `~/cascadia-cluster`

`orchestrator.yaml` runs the authoritative scheduler and the small john1 worker.
`compute.yaml` is a template; the installer substitutes the node name and private
Tailscale address. The compute authentication token and MinIO credentials are
supplied by the generated `secrets.env`, never committed.

The advertised `Disk` capacity is transient execution scratch, not artifact
storage. Jobs must publish durable outputs through Bacalhau `S3Managed` result
paths, normally `/outputs`, and treat `$CASCADIA_SCRATCH_ROOT` plus every other
worker-local path as disposable after the execution exits.

Bacalhau is pinned to v1.9.0. The verified Darwin arm64 binary SHA-256 is
`adb62f07b9e0ef2122f11714ba9bc233c8a4e36d61b4044603c7dbea638bd7c7`.

The orchestrator uses a 10-second scheduler retry backoff and a 1,000-delivery
evaluation-broker budget. These protect transient scheduler/evaluation work;
they do not enlarge Bacalhau v1.9's finite over-subscription queue. Large maps
therefore use the durable `cascadia_cluster` scheduler-capacity admission
window: all logical items are recorded up front and the client releases only
the aggregate number that connected capacity can pack. Bacalhau still owns
placement, admission of each released job, retry, and rescheduling. The client
never creates host batches, affinity, or partitions.
