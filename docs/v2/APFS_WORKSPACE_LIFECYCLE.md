# R2-MAP APFS workspace lifecycle

Status: retired before execution by ADR 0195. No sparsebundle was created,
attached, mounted, detached, or executed. Everything below documents the
historical John1-SSD design and must not be used for new campaign work.

The active canonical root is
`john2:/Users/john2/cascadia-bench/r2-map-v1` on John2's native internal APFS
Data volume. Do not run this document's mutating commands and do not create or
touch the listed `/Volumes/John_1` paths. New build/cache/temp paths are
allocated under the John2 root. John1 receives verified training windows and
model/checkpoint payloads only in memory. Its sole executable exception is the
ADR 0195 phase-ephemeral generation runtime: one signed/hash-verified arm64
binary plus <=64-KiB manifest in a registered `/private/tmp` directory, <=64
MiB combined, with streamed output and mandatory cleanup. No cache, build tree,
dataset, checkpoint, log, replay, or result may accompany it.

## Frozen identity

- Backing: `/Volumes/John_1/cascadia-cluster/r2-map-v1/storage/r2-build.sparsebundle`
- Mountpoint: `/Volumes/John_1/cascadia-cluster/r2-map-v1/apfs-work`
- Marker: `apfs-work/.r2-map-apfs-workspace.json`
- Volume: `CascadiaR2Build`, APFS, 64 GiB
- Campaign allocation: 40 GiB
- Backing SSD free floor: 140 GiB
- Mounted workspace free floor: 40 GiB
- Direct work children: `cargo-target`, `tmp`, `cache`
- Owner/group: current campaign owner; directory mode: `0700`

`python/cascadia_mlx/r2_map_apfs_workspace.py` is the pure identity authority.
`r2_map_apfs_lifecycle.py` adds transactional operation semantics without
weakening that validator.

## CLI

```bash
PYTHONPATH=python .venv/bin/python tools/r2_map_apfs_workspace.py plan
PYTHONPATH=python .venv/bin/python tools/r2_map_apfs_workspace.py status
PYTHONPATH=python .venv/bin/python tools/r2_map_apfs_workspace.py verify
```

These commands do not mutate mount state. Mutating commands require the
explicit `--execute` flag:

```bash
... r2_map_apfs_workspace.py create --execute
... r2_map_apfs_workspace.py attach --execute
... r2_map_apfs_workspace.py detach --execute
... r2_map_apfs_workspace.py recover --execute
```

`create` and `attach` additionally require a valid `control/host-safety.json`
whose 60-second quiet window passed, `syspolicyd` remained at or below 256 MiB,
and system swap did not grow above its baseline. The current receipt is
`blocked-host-recovery`, so those operations fail before invoking `hdiutil`.
Detach and recovery remain available to safely unwind a journaled mount during
a host stop.

## Transaction and recovery protocol

Every mutation takes an exclusive lock and writes/fyncs
`control/apfs-workspace-operation.json` before invoking an exact absolute-path
command. The journal records operation, stage, exact backing/mount paths,
device when known, and command identity. The latest compact status is atomically
written to `control/apfs-workspace-status.json`.

- Create is idempotent when the exact backing directory already exists.
- Attach requires the exact empty mountpoint and immediately observes
  `hdiutil info -plist` plus `diskutil info -plist`.
- The observation must match backing, mountpoint, volume name, canonical UUID,
  APFS, 64-GiB capacity, free-space floors, owner/group, `0700`, read-write
  state, and no symlink components.
- Attach creates only the three direct work children and an atomic `0600`
  marker after identity passes. A mismatch on an owned just-attached image is
  immediately unmounted/detached; a foreign occupant is never detached.
- Verify is read-only and revalidates marker plus live observation.
- Detach revalidates identity, unmounts the exact mountpoint, detaches only the
  observed `/dev/disk*` image device, and requires the observation to vanish.
- Recover completes create, marker publication, or detach according to the
  hash-checked journal. Without a journal it never adopts an invalid or markerless
  foreign mount.

Fault-injection tests cover interruption before/after every create, attach, and
detach boundary, idempotent repeats, identity-mismatch unwind, foreign-mount
refusal, and host-stop enforcement. Tests use an injected command runner; they
never call host mount tools.

## Preflight and environment

After a separately authorized mount, `preflight_storage` must receive the
authoritative spec and collected observation. It independently requires a
distinct mount device, exact ownership/mode, nonsymlink containment, and:

```bash
TMPDIR=/Volumes/John_1/cascadia-cluster/r2-map-v1/apfs-work/tmp
CARGO_TARGET_DIR=/Volumes/John_1/cascadia-cluster/r2-map-v1/apfs-work/cargo-target
UV_CACHE_DIR=/Volumes/John_1/cascadia-cluster/r2-map-v1/apfs-work/cache
```

Atomic rename/fsync is probed on APFS itself. No internal-disk or raw-ExFAT
execution fallback is accepted.

## Dashboard host blocker

`control/host-safety.json` is the only live canonical mutation made while the
host is stopped. Dashboard construction overlays it onto John1's existing host
receipt without changing campaign phase or phase intent. The UI therefore sees
`intent=control` and detail beginning `blocked-host-recovery`, with current
`syspolicyd` RSS, the 4-GiB hard stop, 256-MiB recovery threshold, swap baseline,
and observed swap. Scientific state remains revision 0, `contracts-ready`.
