# R2-MAP W4 external Mach-O Gatekeeper incident

Status: hard host stop-rule trip; resource panels invalid and pending recovery.

## Scope and scientific impact

W4 protocol, model-identity, exhaustive-selection, and lightweight cross-language
tests completed. The full 80-turn heterogeneous runner smoke did not complete,
and the real width-192 maximum-width resource panel is not accepted. Neither
failure changes game data, checkpoints, model weights, or promotion state.

The deterministic protocol fixture and real checkpoint both returned complete,
ordered, finite 6,372-action outputs with zero pruning and zero process swaps.
The real checkpoint call took 5.354067834 seconds at 1,254,653,952 bytes peak
process RSS. Its system-swap delta was 2,043,475,394 bytes, however, while an
unrelated macOS Gatekeeper scan was consuming the host; this makes the resource
panel invalid rather than passing.

Evidence:

- `/Volumes/John_1/cascadia-cluster/r2-map-v1/smoke/r2-map-max-width-real-mlx-v1.json`
  (SHA-256 `5e2155344ba1e6ac080b0f790b88e3ccdb9657086692a087e19fb1cc5f456d7e`)
- `/Volumes/John_1/cascadia-cluster/r2-map-v1/smoke/r2-map-max-width-protocol-fixture-v1.json`
  (SHA-256 `2424260b09658cccecdbf15568da359a744979dfe50958a68617023565dd6611`)

## Timeline and containment

- A debug `cascadia-search` test executable was built only under the campaign
  SSD target and used to begin the four-model, 80-turn correctness smoke.
- The test itself remained CPU-bound and below 0.8 GiB RSS, but
  `/usr/libexec/syspolicyd` grew continuously while scanning the external-volume
  Mach-O. The test was aborted at approximately 14 minutes 55 seconds before
  the 4 GiB per-process stop threshold.
- The exact generated debug test executable was removed. `syspolicyd` continued
  processing the deleted vnode.
- A release test build subsequently completed on the SSD. A stable copy was
  explicitly ad-hoc signed and passed `codesign --verify --strict`, but it was
  never executed: `syspolicyd` crossed the stop threshold during copy/sign.
  Both generated external test executables were immediately removed.
- At `2026-06-18T05:51:25Z`, `syspolicyd` was still at 4,356,368 KiB RSS and
  147.5% CPU. macOS reported 1,907.44 MiB swap used. No agent attempted to kill
  or restart the system daemon.

All Python commands used `PYTHONDONTWRITEBYTECODE=1`. All W4 build and temporary
paths after the separately recorded containment incident remained under
`/Volumes/John_1/cascadia-cluster/r2-map-v1/`.

## Root-cause evidence and hypotheses

The observed blast radius belongs to macOS execution-policy handling, not the
R2-MAP model process:

- `syspolicyd` sustained roughly one to one-and-a-half CPU cores and grew from
  about 2.6 GiB to beyond 4 GiB while the external binary was active;
- the test process stayed below 0.8 GiB;
- the debug binary had no reported quarantine xattr and already had a
  linker-generated ad-hoc signature;
- removing the binary did not immediately release the daemon's scan state;
- replacing the release binary signature with an explicit, verified ad-hoc
  signature did not prevent the ongoing scan.

The leading hypothesis is pathological Gatekeeper assessment of large Rust test
Mach-O files on the removable/external volume. `/Volumes/John_1` is ExFAT and
mounted `nodev,nosuid,noowners`, which is the strongest concrete filesystem
explanation for repeated assessment and weak executable identity semantics.
Other hypotheses are accumulated backlog from concurrent external executables
or a macOS `syspolicyd` defect.
The evidence does not justify weakening the swap or per-process stop rules.

The contained permanent direction is a bounded APFS sparsebundle whose backing
and mountpoint both remain under the campaign root:

- backing: `storage/r2-build.sparsebundle`;
- mountpoint: `apfs-work/`;
- capacity: 64 GiB; campaign allocation budget: 40 GiB;
- pre-growth backing free floor: 140 GiB, preserving the existing 100 GiB SSD
  reserve after the full 40 GiB budget;
- mounted APFS free floor: 40 GiB;
- owned mode `0700`, exact APFS volume UUID and marker identity, no symlink
  components, and no internal-disk fallback;
- Cargo target, temp, and cache roots are direct children of `apfs-work/`.

The authoritative no-I/O validator is
`python/cascadia_mlx/r2_map_apfs_workspace.py`, with adversarial path and mount
observation tests. No sparsebundle was created or mounted during this incident.

## Required recovery preflight

No full-game or resource panel may resume until all conditions pass:

1. The OS or user has restored `syspolicyd` to an idle/restarted state. Agents
   must not kill the system daemon.
2. During a 60-second quiet window, `syspolicyd` remains below 256 MiB RSS and
   below 5% CPU, process swap counts remain zero, and system swap usage has zero
   positive delta.
3. A stable release artifact on the SSD has a recorded SHA-256, no quarantine
   xattr, and a verified signature. Before using raw ExFAT again, the APFS
   sparsebundle must be created and mounted by a separately authorized
   operational tool. Attach is transactional: exact empty mountpoint, immediate
   `hdiutil`/`diskutil` APFS and UUID verification, marker/owner/mode/capacity/
   free-space verification, and immediate detach on any mismatch. Creating or
   assessing the release must not violate the same quiet-window gate.
4. The verified width-192 checkpoint maximum-width panel is rerun first. It must
   preserve all 6,372 identities and finite outputs, prune zero actions, use no
   remote inference, keep every process below 4 GiB, report zero process swaps,
   and show zero positive system-swap delta.
5. The four-model 80-turn release smoke then runs under continuous monitoring
   with the same 4 GiB/process and zero-swap-growth hard stops.
6. Reports, binaries, targets, and temporary files remain inside the campaign
   SSD tree, with executable build/cache work inside the verified APFS mount.
   Unmount uses a clean `diskutil unmount`/`hdiutil detach`; recovery first
   identifies the exact image device and refuses foreign occupants or mounts.
   John4 remains out of scope.

Until that preflight succeeds, the correct status is `pending-host-recovery`,
not pass, fail, or scientific regression.
