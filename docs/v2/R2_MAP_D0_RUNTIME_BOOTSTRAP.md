# R2-MAP D0 runtime bootstrap

This runbook is the executable infrastructure contract for
`d0-runtime-bootstrap-20260618-v1`. [`CASCADIA_V2_GOAL.txt`](../../CASCADIA_V2_GOAL.txt)
is the single source of truth; this runbook supplies subordinate D0 procedure
and cannot override the goal.

John1's internal `/Users/johnherrick/cascadia-bench/r2-map-v1` root is the sole
primary active store and result-install authority. John2 remains the only
buildx/OCI builder and holds only dependency-closed old-research archives.

Status on 2026-06-18: **project execution blocked**. All three hosts are sealed,
live, and independently qualified at 10 vCPU/14 GiB for their current roles;
John2 alone has buildx. Their bounded role receipts are installed under John1's
active root. John3's pre-existing native workspace is frozen, archived on
John2, and independently reopened by John1 with every manifest entry and
hardlink verified. The separately authorized cleanup deleted the exact 7,052
manifest entries (978,085,576 unique regular bytes); John1 independently
rechecked that source, staging, and quarantine are absent and that John3's
execution runtime is unchanged. The obsolete John2 rendezvous, John2-canonical
publication, peer-key, and John2-to-John3 transport source has been removed.
D0 remains RED until superseding live role receipts, the complete two-cycle
graph, every source gate, and the signed topology aggregate all pass.

## Fixed topology and storage

| Host | Role | D0 authority |
|---|---|---|
| John1 | orchestrator, sole source tree, primary active-storage owner, offline signer, result installer/aggregator, native MLX/Metal host, execution worker | render/sign exact packets; install the execution-only runtime; run no OCI build; accept build and worker bundles; publish canonical control/dashboard state |
| John2 | sole acquisition/build host, execution worker, cold archive owner | acquire frozen inputs; install buildx; run the non-project BuildKit probe; return build evidence and OCI bytes to John1; return result bundles; archive only dependency-closed old research |
| John3 | execution worker | materialize only John1-accepted image/input bytes; install no buildx; run non-build verification; return result bundles directly to John1 |
| John4 | excluded | none |

All source edits happen in `/Users/johnherrick/cascadia` on John1. All active
authoritative artifacts, bundles, receipts, aggregate state, and dashboard
inputs live under John1's owner-private internal APFS root:

```text
/Users/johnherrick/cascadia-bench/r2-map-v1
```

John2 and John3 retain only bounded local runtime, VM, image, cache, builder or
job staging required by their role. Those objects are non-authoritative and are
removed by qualification/lease cleanup. John2's separate cold root is
`/Users/john2/cascadia-bench/r2-map-archive-v1`; it accepts only old objects via
the goal's manifest/checksum archive transaction and is never an active
fallback. The retired `/Volumes/John_1` SSD and John4 are never sources,
destinations, caches, or fallbacks.

Each host uses native arm64 Apple VZ with 10 vCPU, 14 GiB RAM, a 2-GiB macOS
reserve, a 5-GiB root disk, and a 13-GiB data disk. Host sharing, Rosetta,
binfmt, nested virtualization, Kubernetes, port forwarding, agent forwarding,
and automatic Docker-context activation are disabled. John2 alone has buildx
and OCI build authority.

## Safety invariants

- D0 is infrastructure-only. It cannot import Cascadia project code, read a
  protected seed, build a project image, or run a project command.
- Every mutating operation uses a canonical, time-bounded Ed25519-signed packet
  and the exact `--execute --confirm-...` confirmation.
- Every predecessor is a signed v3 result bundle plus an exact materialization
  receipt. The consumer reopens the bytes before mutation.
- Swap must remain zero for the complete phase and for persistence. Command
  output is capped cumulatively for the phase.
- John1 physical APFS/internal/non-removable/solid-state identity, owner-private
  path, 64-GiB active-root ceiling, and 64-GiB post-operation free-space floor
  are rechecked immediately before every active commit. John2's equivalent
  checks apply only to an explicitly authorized cold-archive commit.
- Result manifests, reports, and every payload are covered by the campaign-key
  signature. Unsigned draft exports are transport-only and never aggregate
  eligible.
- The absent baseline and rollback namespace contain only campaign-owned
  Colima profile/data/cache state, isolated Docker client/buildx state, and
  bounded campaign staging. The exact global Homebrew formulae are positively
  identified, recorded as pre-existing immutable dependencies, and never
  installed, uninstalled, or cleaned by D0.
- Rollback may remove only campaign roots proven absent by the authenticated
  preflight.

## Reviewed helper closure

The entrypoint is `tools/r2_map_d0_runtime.py`. Its deterministic USTAR helper
contains only standard-library Python sources:

```text
r2_map_d0_runtime.py
r2_d0/__init__.py
r2_d0/aggregate.py
r2_d0/artifacts.py
r2_d0/authorization.py
r2_d0/bootstrap.py
r2_d0/bundle.py
r2_d0/canonical.py
r2_d0/cli.py
r2_d0/closure.py
r2_d0/dashboard.py
r2_d0/inventory.py
r2_d0/ingress.py
r2_d0/runtime.py
r2_d0/signing.py
r2_d0/storage.py
r2_d0/transport.py
```

The helper must be invoked with `/usr/bin/python3 -I -S -B`. Packaging parses
imports, rejects external modules, freezes USTAR metadata, and verifies the
complete source manifest. The obsolete
`docs/v2/reports/r2-map-d0-infrastructure-source-v1.json` is explicitly
rejected and cannot authorize execution.

Read-only helper review:

```bash
/usr/bin/python3 -I -S -B tools/r2_map_d0_runtime.py build-helper \
  --source-root tools \
  --out /absolute/review/path/r2-map-d0-helper-v1.tar

/usr/bin/python3 -I -S -B tools/r2_map_d0_runtime.py verify-helper \
  --archive /absolute/review/path/r2-map-d0-helper-v1.tar
```

## Bootstrap and trust edges

### Campaign helper and public key

Root may approve one raw SHA-256 bootstrap packet per host before a campaign
public key exists. `apply-bootstrap` can install only the reviewed helper and
normalized campaign public key at their fixed owner-private destinations. It
validates everything before writing and rolls the pair back atomically on
failure.

```bash
/absolute/review/path/r2_map_d0_runtime.py apply-bootstrap \
  --packet /absolute/path/bootstrap-packet.json \
  --authorized-packet-sha256 <root-published-raw-file-sha256> \
  --helper-archive /absolute/review/path/r2-map-d0-helper-v1.tar \
  --public-key /absolute/path/campaign-public-key
```

Render and sign one bootstrap record for each of John1, John2, and John3. The
final aggregate requires exactly those three records and signatures.

### Direct John1 control edges

John1 is the only control-plane endpoint. It delivers immutable signed work
packets independently to John2 and John3 over their existing administrator
channels and retrieves bounded immutable result bundles directly. John2 and
John3 never install peer credentials, negotiate shared state, or transfer
worker results to one another. A returned bundle becomes authoritative only
after `r2_d0/ingress.py` verifies its signatures and exact archive identity and
atomically installs it beneath John1's active root.

## Frozen non-project inputs and runtime supply

- Colima, Lima, Docker CLI, and John2-only buildx use the exact versions,
  bottle sizes, and SHA-256 identities in `tools/r2_d0/canonical.py`.
- John2's integrated buildx driver must report the unique exact identity
  `Driver: docker`, `BuildKit version: v0.30.0`, and `Platforms: linux/arm64`;
  duplicate labels, alternate drivers, versions, or platforms fail closed.
- Colima-core v0.10.4 is the 332,354,401-byte arm64 Docker image with SHA-256
  `1fc0354f4f99734ce3886628cc7af8b0437c1a1d391b126bd09cba0df35ee53f`.
- Alpine 3.22.1 is selected by index, platform manifest, config, and layer
  digest; every object size is checked before deterministic OCI rendering.
- The BuildKit scanner source, license, OCI descriptors, SPDX output, and
  maximal provenance are independently bound.
- The nftables egress policy is passed as one validated bounded ASCII argv
  value and piped to `nft` inside the guest; Colima's host-side stdin remains
  closed, avoiding a transport deadlock while preserving atomic policy load.

John2 runs, in signed order:

```text
acquire-core
acquire-homebrew-artifacts
acquire-scanner
acquire-smoke
render-runtime-supply
render-probe-context
install
```

`render-runtime-supply` creates one deterministic USTAR containing the exact
Colima core, Alpine OCI, and worker-role Homebrew closure. This replaces the
retired per-artifact staging surface.

- John2 returns the rendered supply and build evidence to John1. John1 verifies
  and accepts those exact bytes once, then independently distributes the
  accepted supply to the John1 and John3 execution stores. No peer-to-peer
  worker channel is permitted.

## Packet rendering and execution

Canonical packet/signature commands are:

```bash
/usr/bin/python3 -I -S -B tools/r2_map_d0_runtime.py render-packet \
  --kind work --spec /absolute/path/spec.json --out /absolute/path/packet.json

/usr/bin/python3 -I -S -B tools/r2_map_d0_runtime.py sign \
  --payload /absolute/path/packet.json \
  --private-key /owner-private/path/campaign-ed25519 \
  --out /absolute/path/packet-signature.json

/usr/bin/python3 -I -S -B tools/r2_map_d0_runtime.py verify-signature \
  --payload /absolute/path/packet.json \
  --signature /absolute/path/packet-signature.json \
  --public-key /absolute/path/campaign-public-key
```

Every signed phase command also receives `--packet`, `--signature`, and
`--public-key`; every mutation additionally receives `--execute` and the exact
packet SHA-256 confirmation. `plan` is read-only and emits only the frozen argv
plan.

## Two-cycle D0 sequence

Run each healthy host independently as soon as its required inputs exist. Do
not add a three-host barrier to host-local installation. The only global
barrier is before Cascadia project execution.

1. **Qualification preflight** — John1, John2, and John3 each run `preflight`
   with operation `preflight-audit`. Freeze campaign-runtime absence; positively
   hash and version the role-specific global Homebrew dependencies; and record
   the exact platform, internal storage, memory pressure, zero swap,
   listener/process/mount absence, and the 20-GiB plus 25%-of-free ceiling.
   The global dependency probe uses no campaign cache, log, temp, Colima, or
   Docker-client environment path, so the probe cannot recreate a root after
   the absent-baseline snapshot.
   John1 also proves the post-cleanup Podman machine/storage/activity absence.
2. **Acquisition** — John2 acquires core, Homebrew closure inputs, scanner, and
   Alpine, then returns the supply and probe context to John1. Acquisition
   packets use progressive artifact identities: an output may remain null only
   through the packet that produces it, while every earlier completed output
   is bound by its sealed predecessor. This leaves a legal first acquisition
   packet without pretending future derived artifacts already exist.
3. **Worker materialization** — John1 accepts the exact supply once and sends
   identical accepted bytes independently to John1 and John3. Every consumer
   reopens its signed input identity.
4. **Install** — all three hosts run `install` with operation
   `install-runtime`. The packet-pinned bottle closure is verified, but the
   already-qualified global formulae are not mutated. John2 positively verifies
   the frozen buildx dependency; John1 and John3 continue to reject buildx.
5. **Start** — all three run `start` with operation `start-runtime` against byte-exact config and
   inputs.
6. **Verify** — all three run `verify-runtime`. Validate Engine identity,
   effective disks/config, no unexpected host or guest listener, no shared
   mounts/Rosetta/KVM/Kubernetes, hardened no-network/non-root smoke execution,
   complete cleanup, and identical stop/start recovery. John2 additionally
   runs the integrated Docker-driver BuildKit feature probe and proves semantic
   cache/payload emptiness before and after. The sealed daemon configuration
   enables Docker's containerd snapshotter and explicitly selects `cgroupfs`;
   the corresponding verified Engine identities are therefore `overlayfs` and
   `cgroupfs`. Engine comparison treats only Docker's documented unordered
   `RegistryConfig.InsecureRegistryCIDRs` collection semantically: it must be
   the unique exact set `127.0.0.0/8` and `::1/128`, then is sorted before
   hashing. Registry keys, mirrors, the `docker.io` index configuration, and
   every other Engine identity field remain exact; duplicates, extras, wrong
   CIDRs, mirrors, or index values fail closed.

   Guest package/license evidence is a total one-to-one inventory: every dpkg
   package has exactly one canonical copyright-path record. Present documents
   require an exact positive byte count and SHA-256; packages that genuinely
   ship no document are retained as explicit `exists=false`, `present=false`,
   `size=0`, `sha256=null` records and remain SPDX `NOASSERTION`. Symlinked
   Debian shared-document paths are admitted only when the requested-path
   symlink and resolved canonical document path are both explicit. Missing or
   duplicate records, unknown packages, unrecorded aliasing, path escapes, and
   false presence or absence fail closed. The report distinguishes a complete
   inventory from whether every package shipped a document. Docker's Ubuntu
   `containerd.io` package is also proven by exact dpkg ownership of both
   `/usr/bin/containerd` and `/usr/bin/runc`.

   Guest TCP listeners are checked against the guest's own exact `ip -j
   address` projection. The only interfaces are `lo`, `eth0`, and `docker0`;
   loopback, the fixed `192.168.5.1/24` address, `eth0`'s single link-local IPv6
   address, and `172.17.0.1/16` are validated before deriving DNS endpoints.
   SSH may listen only on its expected wildcard port 22 endpoints; DNS may
   listen only on loopback or the exact `eth0` addresses. Wildcard DNS,
   `docker0` DNS, extra interfaces, and any address not derived from this
   projection fail closed.

   Binfmt classification is semantic, not name-count based. The frozen guest's
   exact native `python3.12` handler is admitted only with interpreter
   `/usr/bin/python3.12`, empty flags, offset zero, and magic `cb0d0d0a`;
   `register` and `status` are the only control entries. Rosetta, QEMU,
   foreign-architecture, altered native, additional, and unknown handlers fail
   closed.

   Nested-virtualization capability is checked independently from the kernel's
   compiled-in KVM module. The frozen arm64 guest must expose the exact generic
   `/sys/module/kvm` identity but no `/dev/kvm`, no architecture-specific KVM
   modules, and therefore no usable nested-virtualization device. A usable
   device or any module-set drift fails closed.

   Colima creates exactly one isolated named Docker context,
   `colima-cascadia-r2`; verification requires only that context plus the
   unchanged built-in `default`, exact owner-local metadata and dedicated
   socket endpoint, no TLS or credential material, and effective context
   `default` under the packet-pinned `DOCKER_HOST` (automatic activation stays
   disabled). Host activity is also role-exact: one owner-local Lima usernet,
   one owner-local hostagent, and one SSH mux with their full packet-derived
   argv and required Unix sockets; only usernet's loopback ephemeral listener
   and hostagent's deterministic DNS listener are admitted. John1's required
   dashboard heartbeat can overlap this snapshot, so the gate separately
   admits only its exact read-only `colima status` -> `limactl list` observer
   chain when it descends from the exact init-owned, owner-local dashboard
   watcher argv, including macOS's brief pre-exec `(colima)` / `(limactl)` argv
   forms. Wrong profiles, orphaned children, altered watcher arguments, and
   mutating or unknown runtime clients fail closed, as does any extra
   init-owned process, listener, socket, launchd item, or host mount.
   The same authenticated read-only observer chain is admitted while proving
   the runtime inactive immediately after an exact stop or delete; it does not
   count as daemon activity, while any runtime daemon, socket, listener,
   launchd item, mount, or unauthenticated process still fails the inactive
   gate.
   Immediately after stop/start recovery, activity must converge within a
   fixed five-second deadline and 200-ms sampling interval to two consecutive
   identical valid stable projections. Authenticated dashboard observer
   children are recorded but excluded from the stable daemon projection;
   invalid samples reset the consecutive count, and persistent extras or
   alternating valid instability fail closed. Failure evidence retains the
   bounded normalized rows, row-set digests, and complete sample sequence.
   Smoke containers omit Docker's invalid literal `--pid private` flag and
   instead prove the effective `HostConfig.PidMode` is exactly empty, which is
   Docker's default-private namespace. `host`, `container:<id>`, the literal
   `private`, missing, and otherwise unexpected PID modes all fail closed;
   network, read-only root, capability, privilege, IPC, cgroup, tmpfs, user,
   and mount limits remain independently bound by the inspect contract.
   Colima 0.10's successful modern `status --json` is statusless, so running
   state is established by the command's zero exit plus an exact effective
   field contract: profile display name, Virtualization.framework driver,
   ARM64 architecture, Docker runtime, virtiofs mounts, packet-derived Docker
   and containerd sockets, Kubernetes disabled, 10 CPUs, 14 GiB memory, and
   13 GiB data disk. Legacy explicit `running`/`started` status is accepted
   only with the same exact effective fields; stopped/error states and missing,
   extra, or incompatible values fail closed.
   OCI smoke and scanner loads use containerd-snapshotter identity semantics:
   the logical Engine image ID is the OCI manifest digest, which transitively
   binds the verified config digest and layer diff ID. Duplicate CLI display
   rows are normalized only when every row is the same single logical image;
   cleanup removes by manifest digest and must restore the exact empty image
   store baseline. Named volumes are inspected independently, then normalized
   to the exact two-name set before use. Any nonzero subprocess report binds
   its complete argv, exit code, bounded stderr preview, and stderr digest so a
   fail-closed result is attributable without replaying mutations.
7. **Qualification rollback** — all three run `rollback` with operation
   `rollback-runtime`, then `postflight` with operation `postflight-audit`.
   Remove only authenticated campaign-owned absent-baseline roots and the
   runtime supply. Re-hash the global formulae and require exact preflight
   identity stability; never uninstall them.
8. **Final-live cycle** — repeat preflight, acquisition,
   materialization, install, start, and verify. Do not roll back. A successful
   qualification cycle alone never makes D0 green.

The exact aggregate graph contains 46 host transactions across the two cycles.

## Draft, seal, materialize, and aggregate

Each host phase first writes only a bounded local
`pending/<report_sha256>` transaction containing the signed packet, packet
signature, and report. A pending transaction is not canonical evidence.

1. John2 and John3 independently close and verify their local drafts, then
   return the exact bounded bytes directly to John1.
   John1 materializes its own draft locally through the same validator.
2. John1 range-reads and verifies the declared size/hash for each draft, runs
   `render-result-manifest-from-draft`, signs that manifest, and runs
   `seal-draft-result-bundle`; `verify-result-bundle` must pass.
3. John1 installs the sealed result and storage receipt through a same-volume
   transaction beneath the primary root and independently reopens the result.
4. John1 creates the target-specific materialization receipt and distributes
   only exact required predecessor/input bytes to John2 or John3.
   Same-host predecessors are installed under `receipts/<report-sha256>`.
   Cross-host predecessors are installed under
   `dependencies/<source-host>/<report-sha256>`; the signed transfer
   authorization, target receipt, successor packet, and runtime reopen must
   all bind that exact namespace.
5. Only after the sealed bundle and John1 install receipt exist may a worker
   delete its pending state or the next packet bind that predecessor.

The implementation must expose named, signed operations for this direct return
flow. The obsolete John2 rendezvous/install operations cannot be aliased or
silently redirected.

`build-final-aggregate` and `sign-final-aggregate` require all 46 sealed
bundles, all materialization receipts, the three signed bootstrap records, and
the exact live topology receipts: three runtime roles, John2 cold archive,
legacy-dashboard termination, John2 archive commit, John1 archive reopen, and
John3 cleanup. `verify-final-aggregate` recomputes the complete graph. The
signed aggregate is atomically installed and reopened on John1; there is no
secondary publication authority or terminal remote-publication record.

Only after this closure may the authoritative dashboard mark D0 green. The
dashboard remains [http://100.110.109.6:5187/cluster](http://100.110.109.6:5187/cluster).

## Cleanup evidence retained from John1 storage recovery

The earlier user-authorized Podman-machine cleanup is non-campaign retention
evidence and cannot become an execution source:

- overall receipt SHA-256:
  `fada8246a7961dec72f2b96b9b70d4220c9f76df2b42f1aa6b494f800813b363`
- exact 16-line inventory SHA-256:
  `944dd7184fc0c795de74fd70bb75a7fcd4e7d12fa26746ea1ac31623eed2c9ce`
- locked-archive reopen SHA-256:
  `eec3d4cd85526471a177341277d5df771c01fed6b22b98a3bdc7f8db2312ab6a`
- retention addendum SHA-256:
  `21539e62cad908a039ba4c975fa741310413a2732e9e45d3ec6cd316d8b48251`

## Failure and recovery

- Preserve the exact failure report. A failed phase may transition only to the
  signed rollback chain.
- Never delete an unreceipted object to make room and never use broad
  `brew cleanup` or an unscoped runtime reset.
- Any signature, lineage, materialization, storage, swap, listener, disk,
  engine, BuildKit, guest, cleanup, or dashboard mismatch keeps D0 red.
- Never substitute another version, tag, host, path, transport, native remote
  project command, John3 build, registry BuildKit daemon, host mount, John4, or
  the retired SSD.

## Local source gate

```bash
.venv/bin/ruff check \
  tools/r2_d0 tools/r2_map_d0_runtime.py tools/r2_d0_test_support.py \
  tools/test_r2_d0_*.py

/usr/bin/python3 -m compileall -q \
  tools/r2_d0 tools/r2_map_d0_runtime.py

PYTHONPATH=tools .venv/bin/pytest -q tools/test_r2_d0_*.py
```

Any source edit invalidates the recorded test count and all candidate hashes
until the complete gate is rerun.
