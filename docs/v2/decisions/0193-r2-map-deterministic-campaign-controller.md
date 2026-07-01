# ADR 0193: R2-MAP deterministic campaign controller

Status: accepted for W6 infrastructure; storage location superseded by ADR
0195; W7 execution remains unauthorized

Date: 2026-06-18

## Decision

R2-MAP will use the repository's existing research queue and experiment ledger.
`tools/r2_map_expert_iteration.py` is a deterministic adapter around those
systems, not another scheduler.

Each phase materializes versioned immutable work packets. A packet binds the
hash-chained campaign state, exact host and phase intent, task kind, command,
dependencies, artifact root, seed lease, idempotency key, retry limit, resource
ceilings, and required scientific gates. Queue tasks are derived from packets
and remain compatible with exactly one of John1, John2, or John3. John4 is
rejected recursively from every R2-MAP packet and receipt.

Workers retain fixed seed leases across retries. Lease expiry cannot move work
to another host. John1 is the only central writer and imports compact receipts
only after the corresponding existing-queue task completes. Imported receipts
must prove a contiguous used seed prefix, immutable artifact hashes, successful
identity/replay/checkpoint/score gates, RSS at most 4 GiB, zero process swaps,
and zero system swap growth relative to a fresh quiet-window baseline.
Pre-existing system swap is recorded but does not fail the gate unless it grows.

The campaign state is the phase authority. Queue, ledger, and dashboard data
are validated projections. Reconciliation repairs missing ledger/dashboard
projections, but rejects queue task drift or receipt drift. Recovery after a
state-CAS interruption deterministically reconstructs the same packet set and
queue DAG. Three consecutive rejected/inconclusive candidates or a retry,
identity, replay, training, memory, or swap failure writes a durable stop.

## Consequences

- Training cannot begin until every generation receipt and the central
  generation aggregate are complete.
- While John1 trains, John2 and John3 have benchmark-only tasks; no generation
  packet exists.
- Candidate gates begin only after both the verified training checkpoint and
  the longitudinal aggregate exist.
- Completed receipts are immutable and idempotent; stale claims and partial
  projections are recoverable without deleting evidence.
- W6 can be tested end to end with isolated state and synthetic receipts
  without modifying the canonical `contracts-ready` campaign. New canonical
  state is on John2 per ADR 0195; prior SSD dry-run evidence remains historical.
