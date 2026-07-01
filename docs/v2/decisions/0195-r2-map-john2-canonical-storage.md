# ADR 0195: John2 canonical storage for R2-MAP

Status: accepted; supersedes the storage-location clauses of ADR 0193 and ADR
0194 and the John1 external-SSD clauses in earlier R2-MAP plans

Date: 2026-06-18

## Context

The campaign was originally registered beneath
`/Volumes/John_1/cascadia-cluster/r2-map-v1`. The user subsequently directed
that R2-MAP storage move off that SSD and onto John2's internal disk. Existing
SSD artifacts contain useful implementation evidence, but continuing to write
the external volume would violate the new storage authority and retain the
launchd/Gatekeeper failure mode already observed there.

## Decision

The only canonical writable campaign root is:

```text
john2:/Users/john2/cascadia-bench/r2-map-v1
```

John2's owner is the filesystem authority for that root. The root orchestrator
remains the logical control plane. A non-John2 host may submit a completed,
immutable, checksummed object over authenticated SSH, but the John2 owner must
verify its expected packet identity and atomically install it on John2. There
is no shared mutable network workspace and no per-game synchronization.

Every authoritative dataset, compact index, materialization window, cache,
build tree, checkpoint, model, run log, benchmark aggregate, report, decision
log, controller state, dashboard status, and terminal sentinel lives below the
canonical root. The 80-GiB campaign budget and 40-GiB per-run gate now apply to
John2. The remote physical contract additionally requires at least 100 GiB
free, at most 80 GiB apparent campaign data, and at most 40 GiB for one run.

`/Volumes/John_1/cascadia-cluster/r2-map-v1` is read-only legacy evidence. New
operations may read a legacy object only through a migration manifest that
binds its exact legacy path, byte count, digest, destination, and verification
receipt. No new lock, timestamp, cache, log, status projection, or sentinel may
be written anywhere on `/Volumes/John_1`. The old tree is never a fallback or
recovery source after an object has been migrated and verified.

Host-local inference remains local because it cannot synchronize per move.
John1 is stricter than the other workers: its MLX process receives bounded,
token-verified dataset windows directly into memory and publishes loss and
checkpoint bundles directly to John2. It may not create campaign datasets,
caches, checkpoints, logs, build trees, replays, results, or training temporary
files on either local disk. Its source checkout and the <=64-KiB disposable
dashboard projection are ordinary disk exceptions.

Three-host generation additionally requires one narrowly bounded runtime
exception: during its assigned generation phase, John1 may hold one
John2-built, signed, hash-verified arm64 executable and one <=64-KiB manifest in
a registered mode-`0700` `/private/tmp/cascadia-r2-map-<run-id>` directory. The
executable is mode `0500`; combined size is <=64 MiB. The manifest binds the
work packet, source-freeze transaction and receipt, build receipt, SHA-256 and
BLAKE3, byte count, architecture, signing identity, and exact designated
requirement. The directory and both files have frozen John1 ownership; each file
has link count one. No other entry is permitted. All output streams to John2,
and startup/signal/normal-exit cleanup plus a persisted remote cleanup receipt
and digest are mandatory.
This is phase-ephemeral, non-authoritative, never a recovery source, and never
uses the external SSD.

John3 may receive a work-packet-bound frozen model or executable when required
for its independent benchmark; that copy is non-authoritative, byte-bounded,
and deleted after its John2 receipt verifies.

## Dashboard topology

The canonical compact status is written on John2 at:

```text
/Users/john2/cascadia-bench/r2-map-v1/control/dashboard-status.json
```

John1's API never mounts or writes the campaign root. A fixed, authenticated
SSH fetch reads exactly that file and atomically replaces the disposable mirror
`artifacts/cluster/r2-map-dashboard-serving-projection-v2.json` on John1. The
mirror is at most 64 KiB and binds `john2`, the exact canonical path, canonical
BLAKE3, canonical update time, fetch time, and exact canonical payload. The API
validates all bindings before decoding the status. Missing refreshes become
stale; they never fabricate freshness.

The API rejects the retired v1 SSD serving projection. Historical files remain
readable only as digest-bound migration evidence; they are never a live status
source or serving fallback.

## Recovery and cutover

Cutover is fail closed:

1. stop old SSD publishers and headless writers;
2. inventory legacy artifacts without modifying them;
3. install and verify required objects under the John2 root;
4. rewrite controller and packet identities to the new root under an explicit
   storage-migration decision;
5. publish canonical status on John2 and validate the v2 John1 mirror; and
6. only then authorize W7 or another campaign phase.

An unavailable or full John2 pauses at the next atomic boundary. It does not
authorize John1 internal storage, the external SSD, John3, or John4 as a
fallback.

## Consequences

- The external-SSD sparsebundle/APFS lifecycle is retired for new R2-MAP work.
- Existing SSD evidence remains byte-preserved and scientifically usable after
  explicit digest-bound import.
- Storage coordination occurs only at immutable artifact and phase boundaries.
- John2 is both a compute worker and storage owner; work packets must reserve
  disk I/O so benchmark timing panels are not contaminated by concurrent bulk
  installs.
- W7 remains blocked until storage constants, preflights, controller state,
  dashboard publication, and recovery tests all prove this topology.
