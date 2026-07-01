# R2-MAP John2 Remote Storage Contract

Status: active storage amendment for the R2-MAP expert-iteration campaign  
Canonical host: `john2` (`100.100.43.38`, SSH alias `john2`)  
Canonical root: `/Users/john2/cascadia-bench/r2-map-v1`  
Transport: authenticated OpenSSH plus the content-addressed R2-MAP worker

## Directive

All new R2-MAP persistent data, source snapshots, build outputs, caches,
temporary files, datasets, checkpoints, logs, reports, benchmark artifacts,
opponent pools, and run products live below the canonical root on john2's
internal APFS Data volume. There is no internal-john1 or external-SSD fallback.
No process may mount the root with SSHFS, synchronize it as a mutable tree, or
silently substitute a local path.

The former root
`/Volumes/John_1/cascadia-cluster/r2-map-v1` is predecessor evidence. It is not
a destination for new work. Migration of any predecessor object requires an
explicit content manifest and a verified immutable transaction; broad copying,
cleaning, indexing, or deletion remains forbidden.

Compute ownership does not change:

- john1 performs MLX training from bounded, verified in-memory windows;
- john2 owns authoritative storage and all source compilation/build execution;
- john2 and john3 perform their assigned benchmarks;
- john4 is never used.

John1 may hold model state and bounded dataset windows in memory. It may not
materialize campaign data, checkpoints, logs, caches, build products, or
training temporary files locally. A bounded, disposable dashboard serving
projection is the sole control-plane exception; it is non-authoritative and
limited to 64 KiB. ADR 0195 separately authorizes one phase-ephemeral generation
runtime: a John2-built signed executable plus <=64-KiB manifest under a
registered `/private/tmp` directory, <=64 MiB combined, with no local output and
mandatory cleanup receipt.

John1 MLX packing and training add a stricter runtime boundary: the complete
Python/MLX/Metal/SSH child runs under `sandbox-exec` with all local file writes
denied except `/dev/null`. The launcher records no-follow metadata snapshots of
scoped local write surfaces before and after the child, and a separate
sandboxed publisher writes the compact attestation to John2. Any denied runtime
requirement or snapshot difference fails closed; local temp, cache, history,
SSH control, and log exceptions are not authorized.
The publisher's request id is deterministically
`req-john1-attestation-<first32(attestation_sha256)>`. Consumers derive and
reopen that direct put receipt, require immutable mode `0400` and no previous
object, and checkpoint-bind both the receipt object and semantic receipt hashes.

## Frozen physical identity

The client and remote worker jointly verify:

- SSH alias `john2` resolves to `100.100.43.38` as user `john2` with the
  dedicated `~/.ssh/john2_codex` identity;
- OpenSSH uses batch mode, strict host-key checking, no password or
  keyboard-interactive authentication, no forwarding, and no TTY;
- OpenSSH forces `ControlMaster=no`, `ControlPath=none`, and
  `UpdateHostKeys=no`; it cannot create a multiplex socket or update host-key
  files during John1 execution;
- SSH compression is an explicit transport choice rather than an inherited
  client default. It is off by default for deterministic low-CPU control
  traffic; callers may construct `SshTransport(compression=True)` for measured
  bulk-window transfers without changing object hashes or receipt semantics;
- the remote user is `john2` (`uid=501`, `gid=20`) on host `john2`;
- the campaign root is a real directory, owned by john2, mode `0700`, with its
  frozen device/inode identity;
- every ancestor is a non-symlink directory on the same Data-volume device;
- the Data volume is internal `Apple Fabric` APFS and matches the frozen
  platform/volume identity digest;
- at least 100 GiB remain free, campaign apparent data stays below 80 GiB, and
  each run stays below 40 GiB;
- the complete campaign tree contains no symlink, device crossing, or special
  file; and
- a write, file `fsync`, atomic rename, directory `fsync`, exact reread, and
  cleanup probe succeeds below `control/`.

The identity digest is intentionally compared rather than printed in normal
operator output. A changed machine, volume, root inode, owner, mode, or capacity
fails closed and requires a reviewed contract revision.

## Remote layout

The provisioner owns these mode-`0700` directories:

```text
r2-map-v1/
  control/{bin,locks,transactions,receipts}/
  source/
  build/
  toolchains/
  home/
  cache/{cargo-home,rustup,uv,pycache}/
  tmp/
  datasets/
  checkpoints/
  logs/
  reports/
  benchmarks/
  bundles/
  opponent-pool/
  runs/
```

The remote worker is installed immutably as
`control/bin/r2-map-remote-worker-<sha256>.py`. Every invocation verifies its
own source hash before parsing a command. Every command is canonical JSON with
a unique request identifier, issue time, worker identity, root identity,
operation, arguments, and SHA-256. Every response is a binary frame that binds
its header, optional payload, command, host identity, result, and receipt hash.
Receipts are retained immutably in `control/receipts/`.

## Data operations

The supported client is
`python/cascadia_mlx/r2_map_remote_storage.py`; the operator entry point is
`tools/r2_map_remote_storage.py`.

### Reads

`open_object(relative)` hashes the complete remote regular file and returns an
object token containing its hash, size, inode, device, timestamps, mode, and a
token hash. `read_range(token, offset, length)` returns at most 64 MiB. The
worker checks the token before the read and the open descriptor before and
after it; the client checks the framed payload hash and exact token/range
binding. `iter_object` composes those bounded windows without local files.

Evidence-producing callers use `open_object_with_receipt`,
`read_range_with_receipt`, and `iter_object_with_receipts`. These preserve the
authenticated John2 `storage_receipt_relative` and `storage_receipt_sha256` for
the open and every bounded range. The locator resolves only to
`control/receipts/<request-id>.json`, so an importer can fetch and reverify the
persisted receipt rather than trust a detached digest. The convenience methods
without the suffix return the same verified token or bytes but intentionally
omit evidence metadata and must not be used in campaign work receipts. Lease
acquire, renew, and release likewise return both receipt fields.

This is the training data path. A token is opened once for a frozen dataset
object and batches are decoded directly from verified in-memory windows.

### Atomic files and streams

`put_stream` requires the expected byte count and SHA-256. The client and
worker hash/count independently, the worker writes an exclusive temporary file,
`fsync`s it, applies mode `0400` or `0600`, renames it atomically, and `fsync`s
the parent. Publication is compare-and-swap against either `absent` or an exact
current hash.

`put_unknown_stream` supports headless stdout/stderr without a local temporary
file. It reads stdin to EOF under a caller-declared bound (hard maximum 1 GiB),
hashes/counts independently on both sides, and publishes an immutable file only
after successful EOF and `fsync`. A bound violation leaves no destination.

The production headless supervisor does not attach these sinks with zsh
process substitution: that construct can hide a failed sink behind the Codex
process's exit status. `tools/r2_map_headless_turn.py` owns both sink children,
copies through anonymous pipes, computes each stream SHA-256 locally, drains
bounded diagnostics concurrently, and accepts a turn only after both sink
processes exit zero and return exact object plus storage-receipt identities.
It concurrently watches the remote-lock heartbeat; any heartbeat, pump, sink,
bound, JSON, or identity failure terminates the Codex child and fails the turn
closed.

### Immutable directory/checkpoint transactions

The transaction manifest binds a unique transaction ID, absent target
directory, sorted unique object paths, sizes, SHA-256 values, and manifest hash.
The sequence is:

1. `transaction-begin`: atomically install the manifest below
   `control/transactions/<id>.staging`;
2. `transaction-put`: stream each declared object into the staging tree;
3. `transaction-commit`: rehash every object, add the manifest as
   `.r2-map-transaction.json`, make data files `0400`, manifest-declared
   executables `0500`, and directories `0500`, rename the complete tree to the
   absent target, and `fsync` both parents; or
4. `transaction-abort`: remove only the exact manifest-bound staging tree.

Checkpoints are always immutable transactions. Mutable loss streams and latest
pointers are separate hash-CAS files, so a crash cannot expose a partial
checkpoint or advance a pointer prematurely.

### Locks and remote execution

Lease locks are owner/token/revision bound, serialized under a worker-local
`flock`, and expire after 10–3,600 seconds. Acquire, renew, and release all
produce hashed receipts.

`run_remote` is the only supported build/process launcher. It requires an
absolute allow-listed system executable or a verified executable below the
campaign root, a contained working directory, bounded argv, an allow-listed
environment, contained output paths, and a timeout. The worker supplies
`HOME`, `TMPDIR`, `CARGO_HOME`, `RUSTUP_HOME`, `CARGO_TARGET_DIR`, uv/Python
caches and tool directories below the campaign root. It executes through a
macOS sandbox that denies file writes outside the root. Stdout and stderr are
`fsync`ed and atomically retained below the requested remote output directory.

Campaign control mutations use the distinct `run_controller` boundary, not a
generic remote run and not raw SSH. The executable must be the owner-executable
`tools/r2_map_expert_iteration.py` inside one immutable `source/<freeze>`
transaction; the caller supplies that transaction's manifest SHA-256, the cwd
must be the same freeze root, and every Python import root must remain inside
it. The worker permits only the reviewed non-initialization controller
subcommands. Its sandbox denies network access and filesystem writes except the
isolated run tmp/build/cache and the canonical `control/` subtree. The native
preflight probe uses registered `TMPDIR` on the same filesystem. Controller stdout/stderr and
the authenticated operation receipt are retained below
`reports/controller-runs/`. Genesis/layout initialization remains an explicit
storage-owner operation; `init` is intentionally absent from this execution
allow-list.

Source is uploaded as an immutable transaction below `source/`; builds execute
on john2 and write below `build/`, `cache/`, and `tmp/`. A Mach-O binary is
never copied to John1 except through one of two narrow deployment boundaries:
the ADR 0195 generation-runtime stager below, or the fixed dashboard API
deployment described in the Dashboard section. The generation stager verifies
the immutable build receipt, hashes, arm64 architecture, code signature,
owner, modes, path, and size before execution; all output streams back through
this transport and the runtime is deleted at the phase boundary.

The generation-runtime exception is deliberately smaller than a normal remote
run. John2 signs and publishes one thin-arm64 executable and one canonical
manifest as an immutable transaction. John1 creates one registered
`/private/tmp/cascadia-r2-map-runtime-<packet>` directory at mode `0700` and
stages exactly `runtime-manifest.json` at mode `0400` and
`cascadia-r2-runtime` at mode `0500`. The combined packet is at most 64 MiB and
the manifest is at most 64 KiB. The manifest binds the work packet, source
freeze transaction and persisted receipt, build receipt, executable SHA-256 and
BLAKE3 digests, byte count, architecture, CodeDirectory hash, signing identity,
exact designated requirement, and portable signature-detail digest.

Before each stage, the client scans only that fixed prefix (at most 16
directories). It cleans a stale packet only when it is an exact two-file,
non-symlink, non-special, manifest-valid packet whose executable still matches
both hashes and code signature; otherwise it fails closed for operator review.
The directory and both files must have John1's frozen uid/gid, and both files
must have link count one; wrong-owner files and hard links are rejected.
Execution uses a macOS sandbox that denies networking and all filesystem writes
except `/dev/null`, with `HOME`, `TMPDIR`, and XDG locations mapped to
`/var/empty`. Stdout and stderr stream concurrently into independently bounded,
receipt-authenticated objects on John2. Success requires the process to finish,
both stream threads to terminate, and both John2 storage receipts to verify.
Normal exit, signal/error exit, and stage failure remove the exact registered
packet, `fsync` `/private/tmp`, and publish an immutable cleanup receipt on
John2. The packet is non-authoritative and never a recovery source.

Every mutating client result exposes `storage_receipt_relative` and
`storage_receipt_sha256`. Higher-level work receipts must bind both values
alongside each artifact URI, byte count, and content digest; a content hash
without the corresponding persisted, authenticated John2 storage receipt is
not sufficient publication evidence.

For bootstrap aggregation, the exact dataset transaction is followed by two
immutable sibling objects rather than mutating the committed directory. First,
`<dataset-target>.generation-manifest.json` binds the three generation
packet/work/storage-receipt chains bijectively to every compact shard and the
exact 100,000-game index/transaction commit. The aggregate work receipt's sole
artifact is that manifest and its direct put-file receipt. Second,
`<dataset-target>.bootstrap-phase-barrier.json` binds the generation manifest,
all four phase receipts, and the same transaction/index identities. The
barrier's put request locator is deterministically derived from its semantic
identity before the final barrier hash is computed, so publication is
non-circular and no caller supplies the locator.

### Ephemeral training windows

An exporter may surface exactly two bounded outputs below its isolated
`build/run-<run-id>/` tree: one non-empty `.json` manifest of at most 2 MiB and
one non-empty `.r2map` payload of at most 1 GiB. `open_ephemeral_run_outputs`
returns an authenticated open receipt and immutable object token for each.
John1 consumes the payload only through receipt-bearing ranges and retains the
ordered range receipts in its work receipt.

After both objects have been consumed and their identities recorded,
`prepare_run_cleanup` rehashes the two outputs and freezes a one-hour cleanup
token. The token binds the host/root identity, run ID, both complete object
tokens, and canonical stat inventories of only `build/run-<run-id>` and
`cache/runs/run-<run-id>`. `commit_run_cleanup` holds the worker lock, rehashes
the two outputs, compare-and-swap checks both tree inventories, removes those
two exact trees, `fsync`s their parents, and returns an authenticated cleanup
receipt. The commit is idempotent for crash recovery: a tree already removed by
the same valid cleanup token is reported as such, while any surviving tree must
still match its frozen inventory. Run stdout/stderr and the run receipt remain
under the declared `reports/`, `logs/`, or `runs/` output path. No expanded
window persists after its cleanup receipt verifies.

If a sandboxed command fails before normal run cleanup, its exact
`tmp/run-<id>`, `build/run-<id>`, and `cache/runs/run-<id>` trees may contain
tool-created symlinks. Recovery uses `prepare_failed_run_cleanup` followed by
`commit_failed_run_cleanup`; raw deletion is forbidden. Prepare performs a
no-follow stat inventory of all three trees and issues a one-hour CAS token.
Commit validates every surviving tree before deleting any of them, then
unlinks only entries inside those exact trees without resolving symlink
targets. A changed tree fails before mutation, an external target is never
touched, and replay of the same verified commit is idempotent.

## Dashboard

The canonical dashboard status is
`/Users/john2/cascadia-bench/r2-map-v1/control/dashboard-status.json` and is
published with the 64 KiB `publish_status` operation. The existing dashboard
fetcher may produce only
`artifacts/cluster/r2-map-dashboard-serving-projection-v2.json` on john1. That
projection embeds the canonical host, path, payload, content hash, canonical
update time, and fetch time; it is disposable and never campaign state.

The production dashboard executable is the other narrow John1 deployment
asset. `deploy-dashboard-api` accepts only a <=64 MiB mode-`0500`
`bundles/dashboard-api-*/cascadia-api` object. It verifies the immutable
transaction manifest, expected SHA-256, byte count, object mode, authenticated
open/range receipts, John1 host/uid/gid, and fixed repository destination;
then it uses same-directory `fsync` plus atomic replace at
`target/release/cascadia-api`. It cannot write an arbitrary local path.

## Operator examples

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  .venv/bin/python tools/r2_map_remote_storage.py install-worker

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  .venv/bin/python tools/r2_map_remote_storage.py provision

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  .venv/bin/python tools/r2_map_remote_storage.py preflight

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  .venv/bin/python tools/r2_map_remote_storage.py deploy-dashboard-api \
    --relative bundles/dashboard-api-EXACT_ID/cascadia-api \
    --expected-sha256 VERIFIED_SHA256

# No local log file is created.
some-headless-command 2>&1 | \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  .venv/bin/python tools/r2_map_remote_storage.py put-stream \
    --relative logs/headless/turn-0001.log --max-bytes 67108864
```

Any missing worker, SSH drift, malformed frame, stale token, CAS mismatch,
short/long stream, hash mismatch, symlink, capacity failure, unexpected target,
or receipt mismatch is a hard stop. There is no local fallback.
