# Complete-Action Frontier Observable Bypass V1 Preregistration

Status: **active preregistered**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-observable-bypass-v1`

The authoritative protocol is
`docs/v2/decisions/0095-frontier-observable-bypass.md`.

Reuse ADR 0094's exact embeddings and locally regenerate a deterministic
148-value observable sidecar from action and prior features on every Mac.
Local regeneration is frozen because it is cheaper than another 2.26 GiB
network relay; payload hashes must still agree across all four hosts.

john1 trains the raw linear probe, john2 the raw nonlinear probe, and john3
the embedding-plus-raw combined bypass concurrently. john4 verifies sidecar
identity and cross-replays all saved probes.

The raw linear gate is 60% train recall and 5% exact train sets. Nonlinear
raw-only and combined gates require 80% train recall, 25% exact train sets,
50% validation recall, and 1% exact validation sets. Combined train failure
authorizes a new trunk representation.

No trunk inference, full-network training, sweep, duplicate training, new
teacher compute, sealed test, gameplay, cloud, or external compute is
authorized.
