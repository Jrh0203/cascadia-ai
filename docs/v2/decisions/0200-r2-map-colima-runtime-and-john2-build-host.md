# ADR 0200: R2-MAP Colima runtime and John2 build host

Status: accepted architecture and version selection; D0 bootstrap not yet
executed

Date: 2026-06-18

Supersedes: the John1 OCI-build and three-runtime-host clauses in the 2026-06-18
execution amendment; the native-executable exception in ADR 0195 remains
historical and is not an executable path

## Decision

R2-MAP uses this topology:

- **John1:** sole source-authoring tree, deterministic context renderer,
  dedicated offline signer, disposable dashboard projection, and sole native
  MLX/Metal training/checkpoint-verification host. John1 has no Colima, Lima,
  Docker CLI, buildx, Docker Engine, BuildKit daemon, Docker context/socket,
  selected-stack or R2 Linux VM, image store, builder, image, container, or
  volume. An unrelated Homebrew Podman formula/CLI remains installed but is not
  an R2 execution path. The user-authorized 2026-06-18 storage cleanup removed
  its stopped machine and VM/cache payload; D0 freezes the surviving empty
  skeleton/connection state and no-machine/no-storage/no-activity invariant.
- **John2:** sole OCI build host, canonical artifact/storage owner, and one
  container generation/evaluation worker. John2 receives one immutable signed
  context archive from John1, verifies it without a checkout or host extraction,
  and supplies the exact archive bytes directly to BuildKit.
- **John3:** same-image container generation/evaluation worker. John3 never
  receives source, extracts a context, runs a compiler, or receives an
  authorized build work packet. Its wrapper exposes no build operation.
- **John4:** outside this campaign.

John2 and John3 use a headless, native-aarch64 Docker-compatible runtime. The
buildx plugin is installed and enabled only on John2. John3 has no buildx plugin,
builder profile, signed build packet, or build operation in its wrapper. Its
stock Docker daemon still implements a build API; that capability is
unauthorized and unreachable through the signed wrapper/work-packet surface,
not falsely claimed absent:

| Component | Frozen version or identity | License | Frozen release identity |
|---|---|---|---|
| Colima | 0.10.3 | MIT | Git revision `00f6c297e92a82c04a4ab507db0a61435650d7e8`; Homebrew `arm64_tahoe` bottle SHA-256 `a9dfd1fa0a4aee62fef75974f39f174e4da774f7ba495c43dd0bcc23633381b8` |
| Lima | 2.1.2 | Apache-2.0 | source SHA-256 `23fa5f4621e355236a10200c4e4f61eae9f69c805c57a107247847b51522ab8a`; Homebrew `arm64_tahoe` bottle SHA-256 `b762e573046db099d16a730ac5b0561ad61b823a337d73c0528750ca2d4f9bd6` |
| Docker CLI | 29.5.3 | Apache-2.0 | Git revision `d1c06ef6b41d88d76866aea43c246cd7c63d04fa`; Homebrew `arm64_tahoe` bottle SHA-256 `bc5abed82384f4456e06b53bea84b71b0f6c0f5dbc249c44b727cb8e2b87510c` |
| docker-buildx (John2 only) | 0.35.0 | Apache-2.0 | source SHA-256 `790e4eb0c98da49c60d2c94cebcd3f1658cd7aca3be82093fcb19b9c1d0ac06b`; Homebrew `arm64_tahoe` bottle SHA-256 `bb8a00f55798493e9fa48fedd4b5d4fcb4e1c7b3d20451a97c88015320ae77de` |
| Colima-core guest | v0.10.4 Ubuntu 24.04 arm64 Docker raw image | mixed Ubuntu/Docker/containerd/runc package licenses; colima-core repository MIT | 332,354,401-byte compressed asset; SHA-256 `1fc0354f4f99734ce3886628cc7af8b0437c1a1d391b126bd09cba0df35ee53f`; SHA-512 `32242674b046b5057e60c4aba334b51e3665f05412cda89ed081cc2de153ae5c41f6b105b5c442cbe48d78e2cc21e9ba1950e406b6fb4fc2fd1dd2259240abbd`; D0 freezes a full package/license/SBOM inventory |
| Guest Docker Engine | 29.5.2 expected | Apache-2.0 | Colima-core v0.10.4 `Makefile`; the live client/server API, containerd, runc, and Git commit identities must still match the D0 receipt |
| Integrated BuildKit server | discovered and pinned at D0 | Apache-2.0 | buildx's frozen `docker` driver uses the guest Engine-integrated worker; D0 records its exact version plus OCI/SPDX-SBOM/max-provenance feature probes; the buildx client version is not a substitute |

The Homebrew metadata above is the official formula API snapshot generated on
2026-06-18. Formulae auto-advance; D0 fails closed if a name resolves to any
other version, revision, license, bottle tag, URL, or digest. `brew install`
against a mutable formula name is never sufficient evidence.

The Colima-core image is the exact `arm64 docker` asset embedded by Colima
v0.10.3. Its [embedded image table](https://github.com/abiosoft/colima/blob/v0.10.3/embedded/images/images.txt)
pins the SHA-512, and the [v0.10.4 release asset](https://github.com/abiosoft/colima-core/releases/tag/v0.10.4)
publishes the SHA-256. The guest image's [build recipe](https://github.com/abiosoft/colima-core/blob/v0.10.4/Makefile)
pins Docker Engine 29.5.2. These source statements do not replace a live D0
identity receipt.

## Why John1 is a no-runtime host

The initial 2026-06-18 John1 audit found:

- Mac mini `Mac16,10`, Apple M4, 10 CPU cores (4 performance and 6 efficiency),
  16 GiB memory, native arm64;
- macOS 26.2 build 25C56;
- APFS Data: 239,362,496 KiB total, 186,885,828 KiB used, and only
  14,287,024 KiB available (about 13.6 GiB);
- Homebrew 6.0.1 under `/opt/homebrew`; and
- none of Colima, Docker CLI, buildx, Lima, `~/.colima`, or `~/.docker` present.

The free-space value above is historical. The completed cleanup receipt records
138,219,474,944 available bytes (about 138.2 GB decimal or 128.7 GiB). John1
nevertheless remains a no-runtime host under the permanent trust/topology
decision: source authoring, offline signing, and native MLX are isolated from
OCI build/execution. The topology does not revert merely because space was
recovered.

A subsequent deeper negative-control audit found a pre-existing Homebrew Podman
formula/CLI and a stopped `podman-machine-default` below
`~/.local/share/containers/podman/machine`. That historical tree included a
100-GiB apparent sparse raw disk and about 4.0 GiB of allocated VM/cache bytes.
On 2026-06-18 the user explicitly authorized storage cleanup phases 1-3. Phase
3 removed the stopped Podman machine and those allocated bytes; it did not
remove the formula/CLI. The current filesystem has empty machine skeleton
directories, a 28-byte empty `podman-connections.json`, no machine disk, and no
active Podman/VM process, listener, mount, or container-storage payload.

The cleanup is proven by the non-campaign retention archive receipt
`john2:/Users/john2/cascadia-bench/john1-offload-20260618-v1/control/overall-cleanup-receipt.json`
(SHA-256 `fada8246a7961dec72f2b96b9b70d4220c9f76df2b42f1aa6b494f800813b363`)
and the exact 16-line pre-deletion target inventory
`docs/v2/reports/john1-storage-cleanup-phase3-target-inventory-v1.txt`
(SHA-256 `944dd7184fc0c795de74fd70bb75a7fcd4e7d12fa26746ea1ac31623eed2c9ce`).
The same non-campaign control directory contains the full locked-archive reopen
verification `retention-reopen-verification.v1.json` (SHA-256
`eec3d4cd85526471a177341277d5df771c01fed6b22b98a3bdc7f8db2312ab6a`)
and `cleanup-retention-addendum.v1.json` (SHA-256
`21539e62cad908a039ba4c975fa741310413a2732e9e45d3ec6cd316d8b48251`).
The offload archive is outside the canonical R2-MAP root, is not campaign
authority, and may never become an execution or recovery input. This historical
cleanup does not alter the finding that the selected Colima/Lima/Docker/buildx/
Engine/BuildKit stack and every R2 runtime object are absent.

The cleanup recovered substantial John1 space, but it does not change the
approved topology. Keeping authoring/signing/native-MLX isolated from the OCI
builder avoids mixed trust domains and restores reproducible build epochs.
John1 therefore remains a no-runtime host by architecture, not by the old free-
space measurement. The external SSD is not a fallback. John2 remains the sole
builder and canonical campaign storage owner.

John1's D0 evidence is a negative control. It must prove absence of selected
runtime executables, package receipts, profile/config/cache roots, VM processes,
sockets, Docker contexts, builders, images, containers, and volumes before D1
and again at every build/runtime qualification boundary. It records the
unrelated Podman formula/CLI as disclosure, then gates the exact post-cleanup
negative-control semantics: empty machine skeleton directories, empty
connection/farm maps, and no machine record, disk, socket, process, listener,
mount, or container-storage payload. D0 does not compare against or recreate
the deleted pre-cleanup VM bytes. R2-MAP may not invoke the Podman CLI or create,
start, mount, connect to, label, inspect, prune, upgrade, or delete a Podman
machine.

## Audited host feasibility

The read-only audits found Docker-compatible runtimes absent on all three hosts.
John2 and John3 are Apple M4 arm64 systems with 10 CPU cores, 16 GiB memory,
Virtualization.framework support, and more than 300 GB free on their internal
APFS Data volumes. John2's canonical evidence is:

```text
reports/infrastructure/john2-docker-readonly-audit-v1.json
SHA-256 b81e0ea1cd3d0986c4d54f2d6f1841df6f2c6dace1fe5d2794d71363ac2a116d
```

The observed John1 and John3 facts remain preflight evidence until D0 installs
their signed canonical audit receipts under John2. No project image may be
built from provisional evidence.

The latest read-only seal gate is fail-closed: John1 26.2/25C56 and John2
26.4/25E246 report `Sealed: Broken`; John3 26.5.1/25F80 reports `Sealed: Yes`.
Accordingly D0 is RED with `execution_blocked=true`. Runtime acquisition,
installation, and start must not proceed until all three pass the authenticated-
root/SIP/sealed-volume check. This ADR does not authorize an OS update or reboot.

## Signature bootstrap

D0 has one unavoidable trust-on-first-use edge: workers cannot verify a campaign
signature before the campaign public key is installed. The root orchestrator may
therefore authorize exactly one canonical bootstrap packet per host by
publishing its full content SHA-256 through the control plane. The packet is
strictly typed and may install only:

- the exact standard-library D0 infrastructure helper bytes and SHA-256;
- the OpenSSH Ed25519 public key and SSH fingerprint; and
- the fixed verifier namespace and owner-private destination paths.

It cannot install a runtime, acquire an image, execute a Docker/Podman command,
open a protected seed, or accept arbitrary argv. The worker recomputes the exact
packet and helper hashes before installation and returns an authenticated
bootstrap receipt. Every later D0 packet is signed and verified through the
pinned public key. The final D0 aggregate, including John1's before/after
negative control, is signed by the dedicated John1 private key. The private key
never leaves John1 or enters source, a packet, an image, John2, or chat.

D0 canonical documents, content identities, signatures, and authenticated
receipts require SHA-256. BLAKE3 becomes mandatory at D1. A D0 host without
`b3sum` does not install an unreviewed utility and does not fabricate a BLAKE3
field; its schema omits that field by versioned contract.

## Runtime profile

Use the named profile `cascadia-r2`. The exact reviewed `colima.yaml` is:

```yaml
cpu: 8
memory: 8
rootDisk: 5
disk: 13
arch: aarch64
runtime: docker
vmType: vz
rosetta: false
nestedVirtualization: false
binfmt: false
mounts: null
mountInotify: false
forwardAgent: false
sshConfig: false
autoActivate: false
portForwarder: none
kubernetes:
  enabled: false
```

`mounts: null` is exact and security-relevant. `mounts: []` is rejected because
Colima interprets it as the default writable home mount. QEMU, Rosetta,
cross-architecture binfmt, nested virtualization, Kubernetes, port forwarding,
SSH-agent forwarding, automatic SSH config, default-context activation, and
startup provision scripts are disabled. No brew service or login item starts
the VM. The wrapper always selects the named profile and its owner-only Unix
socket explicitly; it never relies on the current Docker context and never
creates an unauthenticated TCP listener.

The isolated internal-disk roots are:

| Purpose | John2 | John3 |
|---|---|---|
| Colima state | `/Users/john2/.local/share/cascadia-r2/colima` | `/Users/john3/.local/share/cascadia-r2/colima` |
| Colima downloads | `/Users/john2/Library/Caches/cascadia-r2/colima` | `/Users/john3/Library/Caches/cascadia-r2/colima` |
| Docker client config | `/Users/john2/.config/cascadia-r2/docker` | `/Users/john3/.config/cascadia-r2/docker` |
| Profile socket | `$COLIMA_HOME/cascadia-r2/docker.sock` | `$COLIMA_HOME/cascadia-r2/docker.sock` |

The Docker client configs contain no credentials or `currentContext`. John2's
only static extension enables the separately installed buildx plugin:

```json
{
  "auths": {},
  "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]
}
```

John3's complete static client config is `{"auths": {}}`. Its wrapper and work
packet schema reject every build operation, and D0 proves that `docker-buildx`
and every builder profile/cache are absent. The stock daemon's underlying build
API remains present outside that authorized interface and is recorded as such.

The runtime roots are non-authoritative and outside John2's canonical campaign
root. Canonical context archives, OCI exports, SBOM, provenance, manifests,
packets, bundles, checkpoints, logs, and receipts remain beneath:

```text
john2:/Users/john2/cascadia-bench/r2-map-v1/
```

The verified compressed Colima-core image is also a canonical D0 infrastructure
artifact beneath that John2 root. John2 profile creation reads that exact file;
John3 receives one authenticated digest-bound staging copy. Both invoke Colima
with the explicit local `--disk-image` path. After guest creation and live
identity/SBOM verification, Colima's download-cache copy and John3's staging copy
are deleted and receipted. The canonical John2 object remains available for the
clean D2 profile recreations and is outside the non-authoritative runtime-path
ceiling but inside the campaign storage budget.

The exact local core paths are:

```text
john2:/Users/john2/cascadia-bench/r2-map-v1/bundles/runtime/colima-core-v0.10.4/ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz
john3:/Users/john3/.local/share/cascadia-r2/bootstrap/ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz
```

No runtime or campaign byte may be placed on `/Volumes/John_1`.

## Storage and resource ceilings

On each runtime host, both the combined apparent size and the combined APFS
allocated size of Homebrew runtime formulae, Colima profile and core-image
cache, VM root/data storage, Docker images and writable layers, BuildKit
builders/caches, containers, and volumes are at most 20 GiB and at most 25% of
the free internal bytes frozen by D0. The 5-GiB root disk plus 13-GiB Docker
data disk reserves 2 GiB for formulae, profile state, and transient overhead.
The compressed core-image download cache is removed only after guest creation
and after the guest's source-image, Engine, containerd, and runc identities have
been reopened and verified. D0 records and gates both logical/apparent limits
and actual APFS allocated bytes.

Before Homebrew metadata lookup or fetch, D0 records a no-follow, path/type/mode,
byte-count, and SHA-256 inventory of the complete relevant Homebrew
formula state, opt links, download cache, API cache, logs, locks, and temporary
roots. The installer sets isolated `HOMEBREW_CACHE`, `HOMEBREW_LOGS`, and
`HOMEBREW_TEMP` roots beneath the host's D0 infrastructure namespace and proves
that the pre-existing global Homebrew cache/API trees do not change. Rollback
removes only ledger entries created by D0, preserves every pre-existing entry by
hash, and rejects an extra, missing, replaced, or type-changed path. `brew
cleanup` and broad cache deletion are forbidden.

The exact standalone implementation, two-packet derived-artifact flow, command
surface, and rollback sequence are defined in
`docs/v2/R2_MAP_D0_RUNTIME_BOOTSTRAP.md`. BLAKE3 becomes mandatory at D1 as
specified by the signature-bootstrap contract; it is intentionally not
fabricated or installed during D0.

A preflight refuses start or build when either bound would be crossed. Colima
disks only grow; the wrapper cannot silently expand them. If the two clean
offline builds do not fit, D2 fails and requires an explicit storage-contract
amendment. It cannot increase the disk, prune an unreceipted object, use the
external SSD, use John1, or add a remote builder.

The VM ceiling is 8 vCPU and 8 GiB memory. Every project container receives a
stricter packet-specific CPU, memory, `--memory-swap`, PID, tmpfs, output-byte,
and deadline limit. Host preflight rejects memory pressure or positive swap
growth and preserves headroom for SSH, telemetry, the John2 artifact owner, and
the dashboard.

## No host mounts: volume-only data plane

With Colima host mounts disabled, project containers use no bind mount. The
reviewed host wrapper performs this finite state machine:

1. Verify the work packet, execution manifest, signature, accepted OCI archive,
   command profile, and input bundle before creating an object.
2. Create fresh run-ID-labelled input and output Docker named volumes; reject a
   pre-existing name or label collision.
3. Stream a canonical tar of the already validated input through the fixed
   infrastructure import profile into the input volume. Reopen and hash the
   volume inventory from a non-project helper profile.
4. Run the accepted immutable image ID with the input volume mounted read-only,
   the empty output volume mounted read-write, `--network=none`, a read-only root,
   non-root UID/GID, all capabilities dropped, `no-new-privileges`, no socket or
   device, and the packet's resource limits.
5. Stream one canonical output tar through the fixed export profile into a
   bounded host-side validator. Install only a complete manifest-exact bundle.
6. Remove both job volumes and the container and publish an empty-inventory
   cleanup receipt. No result is authoritative until John2 installs that
   receipt.

The import/export helper identity is part of the signed execution manifest. It
cannot pull an image, accept arbitrary shell, read a host path, or use the
network.

## Source and build data plane

John1 renders the manifest-driven strict USTAR context from its sole source
tree, computes SHA-256 and BLAKE3, and signs the context manifest with the
dedicated offline key. Authenticated transport writes the archive once to a
hash-derived pending John2 transaction. John2's infrastructure wrapper verifies
the signature and raw USTAR contract without extracting it and atomically
commits the archive under the canonical root.

John2 feeds the exact committed archive bytes to BuildKit tar-context stdin or
the byte-equivalent BuildKit API stream and accepts each OCI export only from
stdout or the byte-equivalent BuildKit API stream. There is no John2 checkout,
temporary source tree, bind-mounted source, `git clone`, unbound filesystem
export, or host compiler invocation. The build definition and all offline
dependencies are members of the signed archive. Build networking is disabled.

Reproducibility is sequential and clean:

1. Create exact Colima profile epoch A from the pinned core artifact. Freeze
   buildx's default `docker` driver and integrated BuildKit identity; import no
   cache and create no registry BuildKit container.
2. Build and stream-export OCI A, SBOM A, maximal provenance A, build metadata,
   and test evidence.
3. Stop and delete profile A with its complete data disk. Prove the VM, socket,
   context, image store, integrated BuildKit cache, epoch-scoped Docker
   client/buildx metadata, containers, and volumes are absent.
4. Recreate exact profile epoch B from the same pinned core artifact and replay
   the exact context and build profile with no imported cache.
5. Stream-export OCI B and evidence; compare canonical bytes, descriptor graph,
   config, ordered layers, inventories, executables, and command profiles.
6. Stop and delete profile B with its complete data disk and repeat the absence
   proof, including deletion of B's client/buildx metadata.
7. Accept exactly one canonical OCI archive only when A and B match; create a
   third clean execution-only John2 profile and Docker client config and import
   the accepted archive by the same wrapper path used on John3.

The default Docker driver is frozen because it uses the BuildKit integrated into
the pinned guest Engine. D0 must first prove that this exact driver supports OCI
stdout/API export, SPDX SBOM, and maximal provenance using a pinned non-project
scratch fixture. If any feature is missing, D0 fails and the driver choice
requires an explicit amendment. A `docker-container` driver would require a
separately pinned BuildKit image and is not an implicit fallback.

The image is never rebuilt for a host or run. John3 receives only the signed
execution manifest and accepted canonical OCI archive. D4 compares John2 and
John3 known-answer outputs. John1 participates in D5 only, where the Linux
CPU/reference implementation must match the native John1 MLX implementation and
every selectable checkpoint must finish with native Metal reload verification.

## D0 bootstrap gate

This ADR selects the stack but does not itself install it. The bootstrap is
green only when John2 holds one signed aggregate with:

1. canonical preinstall audits for John1, John2, and John3, plus the exact
   root-authorized public-key bootstrap packet/receipt on each host;
2. exact Homebrew formula JSON and bottle receipts matching this ADR, including
   buildx on John2 only and its absence on John3 while acknowledging the stock
   daemon API;
3. pre/post Homebrew formula/opt/cache/API/log/lock/temp inventories, with exact
   newly-created-path cleanup and pre-existing-byte hash preservation;
4. exact core-image SHA-256/SHA-512 and byte-count receipts plus the guest
   package, license, and SBOM inventory;
5. John2 and John3 install receipts and before/after package inventories;
6. exact profile/config hashes, owner/mode checks, and runtime-root inventories;
7. `colima version`, `limactl --version`, `docker version`, `docker info`, Engine
   API, containerd, and runc identities on both hosts; the exact
   `docker buildx version`, `docker` driver, and integrated BuildKit worker
   identity on John2; and buildx/builder-profile/cache absence plus the recorded
   stock daemon API on John3;
8. a John2 non-project scratch feature probe proving tar/API context ingress,
   OCI stdout/API export, SPDX SBOM, maximal provenance, offline operation, and
   complete profile/data deletion and recreation;
9. owner-only Unix socket proof, no TCP listener, no registry credentials,
   unchanged default Docker context, no host mounts, no forwarding, and no
   unapproved process or startup item;
10. a pinned non-project `linux/arm64` OCI smoke loaded without a registry pull
   and executed under the hardened flags on John2 and John3;
11. identical smoke output plus stop/start recovery and exact project-object
   cleanup on both runtime hosts;
12. aggregate runtime bytes below both storage ceilings; and
13. a John1 before/after negative-control receipt proving the selected stack and
   R2 runtime remain wholly absent, the disclosed Podman CLI was not invoked,
   and the frozen no-machine/no-storage/no-activity semantics remain true; and
14. the dedicated Ed25519 signature over the complete D0 aggregate with its
   pinned public-key fingerprint and authenticated John2 publication receipt.

The expected Engine is 29.5.2, but the live identity is authoritative. An
unexpected Engine, BuildKit, containerd, runc, API level, security option, or
feature-probe result is a D0 failure requiring review; it is not normalized away.
No Cascadia/project Dockerfile, image, source context, protected seed, or project
command may enter the runtime during D0. The sole Dockerfile/context exception
is the exact pinned non-project scratch feature-probe fixture named in D0; its
bytes and digests are part of the D0 decision receipt and it cannot import or
execute Cascadia code.

The final John3 authorization proof is deliberately later: D1 closes the wrapper
operation schema, D3 installs the exact wrapper, and D4 proves that it exposes no
arbitrary Docker API or build operation and rejects a host-mismatched signed
build packet. D0 cannot claim evidence from a wrapper that does not exist yet.

## Reviewed command sequence

These commands are the reviewed bootstrap shape, not a record that they were
run. The D0 installer must substitute the target owner home and write every
stdout/stderr, digest, path, and exit status into the signed receipt.

```bash
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_ANALYTICS=1
export HOMEBREW_CACHE="$HOME/Library/Caches/cascadia-r2/homebrew"
export HOMEBREW_LOGS="$HOME/Library/Logs/cascadia-r2/homebrew"
export HOMEBREW_TEMP="/private/tmp/cascadia-r2-homebrew-$UID"
export COLIMA_HOME="$HOME/.local/share/cascadia-r2/colima"
export COLIMA_CACHE_HOME="$HOME/Library/Caches/cascadia-r2/colima"
export DOCKER_CONFIG="$HOME/.config/cascadia-r2/docker"
export COLIMA_PROFILE=cascadia-r2
: "${D0_ROLE:?D0_ROLE must come from the signed host work packet}"
case "$D0_ROLE" in
  john2) export D0_CORE_IMAGE="/Users/john2/cascadia-bench/r2-map-v1/bundles/runtime/colima-core-v0.10.4/ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz" ;;
  john3) export D0_CORE_IMAGE="/Users/john3/.local/share/cascadia-r2/bootstrap/ubuntu-24.04-minimal-cloudimg-arm64-docker.raw.gz" ;;
  *) exit 64 ;;
esac

# Common John2/John3 packages. Fail closed unless formula JSON is the exact
# 2026-06-18 selection above.
brew fetch --force --bottle-tag=arm64_tahoe lima colima docker

# Resolve each cache path with `brew --cache --bottle-tag=arm64_tahoe NAME`,
# verify the three expected SHA-256 values, then install only those verified
# local bottle paths, in dependency order.
brew install "$(brew --cache --bottle-tag=arm64_tahoe lima)"
brew install "$(brew --cache --bottle-tag=arm64_tahoe colima)"
brew install "$(brew --cache --bottle-tag=arm64_tahoe docker)"

# John2 only: fetch, verify, and install the pinned buildx bottle.
if [ "$D0_ROLE" = john2 ]; then
  brew fetch --force --bottle-tag=arm64_tahoe docker-buildx
  brew install "$(brew --cache --bottle-tag=arm64_tahoe docker-buildx)"
fi

# Install the reviewed 0600 colima.yaml and config.json in the isolated roots,
# verify the local pinned core image, then start without UI or auto context.
colima start --profile "$COLIMA_PROFILE" --disk-image "$D0_CORE_IMAGE"

export DOCKER_HOST="unix://$COLIMA_HOME/$COLIMA_PROFILE/docker.sock"
colima status --profile "$COLIMA_PROFILE" --json
docker version
docker info
# John2 only:
if [ "$D0_ROLE" = john2 ]; then
  docker buildx version
  docker buildx inspect --builder default
fi
```

The John3 D0 installer profile omits both buildx commands and proves the
formula/plugin and builder inventory are absent; it records rather than denies
the stock daemon's underlying API. The final signed-wrapper build rejection is
proved at D3/D4, not borrowed by D0. Before setting the isolated Homebrew
variables, the installer
must ledger the pre-existing global formula/opt/cache/API/log/lock/temp trees.
The implementation must verify bottle hashes before `brew install`; command
substitution alone is not a digest check. John2 must configure buildx's
`cliPluginsExtraDirs` before invoking `docker buildx`. `colima update`, registry
login, registry pull, `docker context use`, a global Homebrew cleanup, and a
Docker socket symlink are forbidden.

## Rollback

Rollback is exact and ownership-ledger driven:

```bash
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_ANALYTICS=1
export HOMEBREW_CACHE="$HOME/Library/Caches/cascadia-r2/homebrew"
export HOMEBREW_LOGS="$HOME/Library/Logs/cascadia-r2/homebrew"
export HOMEBREW_TEMP="/private/tmp/cascadia-r2-homebrew-$UID"
export COLIMA_HOME="$HOME/.local/share/cascadia-r2/colima"
export COLIMA_CACHE_HOME="$HOME/Library/Caches/cascadia-r2/colima"
export DOCKER_CONFIG="$HOME/.config/cascadia-r2/docker"
export COLIMA_PROFILE=cascadia-r2
: "${D0_ROLE:?D0_ROLE must come from the signed host work packet}"

colima stop --profile "$COLIMA_PROFILE"
colima delete --profile "$COLIMA_PROFILE" --data --force
# John2 only:
if [ "$D0_ROLE" = john2 ]; then
  brew uninstall docker-buildx
fi
# John2 and John3:
brew uninstall docker colima lima
```

After deletion, the audited no-follow cleanup helper removes only the isolated
Colima, Docker-config, Homebrew cache/log/temp, and other exact paths recorded as
new by the D0 ledger, and only when their owner, device, type, path, and content
identity match. Every pre-existing Homebrew cache/API/log/lock/temp path is
reopened and must retain its original hash; a replaced or missing pre-existing
byte is a rollback failure. The helper does not use a global `rm -rf`, `brew
cleanup`, Docker prune, or delete an unledgered object. Rollback verifies that no
VM/process/socket/context/builder/image/container/volume remains, the default
Docker context is unchanged, no pre-existing formula was removed, and both
apparent and allocated-byte deltas reconcile to the frozen baseline. The
rollback receipt is installed on John2 before D0 can be retried.

## Consequences

- John1 remains protected from accidental VM/image state even after the
  explicitly authorized cleanup recovered disk space.
- John2's internal hard disk is the sole authoritative storage location and the
  sole build location, eliminating context drift and cross-host builds.
- Build and execution remain distinct leases on John2; generation or benchmark
  work never overlaps a build lease.
- Generation, longitudinal benchmarks, candidate gates, D4, and the no-pruning
  performance comparison use John2 and John3 only, with disjoint deterministic
  ranges and no per-game coordination.
- Native John1 MLX/Metal remains a narrow exception and is verified against the
  Linux CPU/reference path before promotion.
- The 20-GiB runtime ceiling may expose that a dependency closure is too large.
  That is a measured architecture failure, not permission to use the SSD,
  silently expand a disk, add a host, or weaken cleanup.

## Primary references

- [Colima configuration](https://github.com/abiosoft/colima/blob/v0.10.3/docs/FAQ.md)
- [Colima v0.10.3 source and embedded images](https://github.com/abiosoft/colima/tree/v0.10.3)
- [Colima-core v0.10.4 release](https://github.com/abiosoft/colima-core/releases/tag/v0.10.4)
- [Homebrew formula API](https://formulae.brew.sh/docs/api/)
- [Docker Buildx build and OCI exporter](https://docs.docker.com/reference/cli/docker/buildx/build/)
- [BuildKit provenance and SBOM](https://docs.docker.com/build/metadata/attestations/)
