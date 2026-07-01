# Technical Debt

No known v2 technical debt is currently accepted.

The oversized command and research modules recorded here during ADR 0078 were
resolved without rebuilding the frozen collector:

- `cascadia-cli-v2/src/main.rs` is now typed parsing and dispatch only;
- command families own their data, model, policy, oracle, and report workflows;
- search owns lookahead, MLX value, ranking rollout, prefilter, prediction,
  policy-improvement, and oracle mechanisms in separate modules;
- simulation separates pattern strategies from finite-market opportunity math;
- large inline test suites live in dedicated child modules.

`python/tests/test_v2_source_structure.py` prevents the CLI entrypoint from
exceeding 300 lines and prevents any active v2 Rust production module from
exceeding 1,500 lines. New debt or unavoidable compromises must be documented
here with cause, proper fix, and blast radius before merge.

## Headless remote LaunchAgent activation

- What: the Bacalhau LaunchAgent is fully installed on john2, john3, and john4, but macOS
  refuses `launchctl bootstrap gui/<uid>` over a headless SSH audit session with
  error 125. Until the next interactive login creates an Aqua bootstrap domain, a
  bounded `run-forever.zsh` supervisor owns the identical `run-node.zsh` process.
  The LaunchAgent entrypoint terminates that fallback before taking ownership, so
  duplicate schedulers cannot survive a login transition.
- Why the right thing was not viable: installing a system LaunchDaemon requires
  administrator authority that is not available to this process, and no logged-in
  GUI bootstrap domain exists remotely. Pretending the LaunchAgent started would
  leave the compute fabric nonfunctional.
- Proper fix: at the next john2/john3/john4 interactive login, verify the staged
  `com.cascadia.bacalhau.plist` loads and `launchd-entry.zsh` removes the fallback.
  Alternatively, install the same plist as a root-owned LaunchDaemon during an
  approved maintenance window, then remove `run-forever.zsh` and this entry.
- Blast radius: workload placement, failure restart, and current runtime behavior
  are intact; the fallback performs exponential-backoff restart. Colima is also
  running on the remote compute nodes, but its Homebrew service cannot be enabled
  from the same headless SSH bootstrap domain. Automatic service startup after a
  full remote-node reboot is not guaranteed until an interactive login or
  LaunchDaemon installation occurs. John1 already runs under launchd.

## Resolved: sparse Colima disk capacity can exceed host free space

The temporary 128 GiB Bacalhau disk shape was retired in favor of 80 GiB
transient Colima Docker disks on all four nodes. The worker-local disk contract
is now scratch-only: durable artifacts must be published through Bacalhau
S3Managed `/outputs` and imported from MinIO, never left on a worker filesystem.
This keeps the advertised capacity below current john1/john4 host headroom and
aligns scheduler capacity with the intended artifact lifecycle.

## Resolved: R2-MAP dashboard publisher process lifetime

ADR 0195 retired the external-volume publisher and its detached-tmux lifetime.
John2 now owns canonical compact status on its internal APFS campaign root;
John1 only fetches a bounded, authenticated, disposable serving projection.
The old removable-volume launchd failure is historical evidence and grants no
permission to restart the publisher, write the external volume, or use tmux as
a production supervisor.

## External-volume Rust Mach-O triggers pathological Gatekeeper scanning

Storage disposition (2026-06-18): ADR 0195 retired this execution path before
the sparsebundle was created. New R2-MAP build trees and canonical artifacts
belong on John2's internal APFS volume; `/Volumes/John_1` is read-only legacy
evidence. The incident remains documented because its measurements explain why
external-volume execution must not be reintroduced.

- What: executing a large Rust test Mach-O from the required campaign SSD made
  macOS `/usr/libexec/syspolicyd` grow beyond the 4 GiB per-process hard stop.
  The test process stayed below 0.8 GiB. Even after aborting and removing the
  executable, the daemon continued scanning the deleted vnode. An explicit,
  strictly verified ad-hoc signature on a stable release copy did not stop the
  already-running assessment.
- Why the right fix was not completed: recovering or restarting a macOS system
  daemon requires OS/user action outside agent authority. Moving build or run
  artifacts to John1's low-space internal disk would violate the campaign's
  storage contract, and weakening the 4 GiB/zero-swap gates would invalidate
  performance evidence.
- Historical proper fix before ADR 0195: after OS/user recovery, establish a
  60-second idle Gatekeeper and
  zero-swap-growth preflight, then create a bounded APFS sparsebundle fully
  contained inside the campaign root. `/Volumes/John_1` is ExFAT mounted
  `nodev,nosuid,noowners`; the APFS image supplies stable executable filesystem
  semantics without internal-disk fallback. Validate its exact backing path,
  mountpoint, UUID, APFS identity, `0700` ownership, 64 GiB capacity, 40 GiB
  allocation budget, 140 GiB backing free floor, and 40 GiB mount free floor
  before placing Cargo/temp/cache roots inside it. If assessment still grows,
  use a properly signed/notarized least-privilege runner or an explicitly
  user-approved removable-volume policy. Re-run both the real width-192
  6,372-action panel and the 80-turn heterogeneous game under continuous
  hard-stop monitoring. The pure static contract is implemented in
  `python/cascadia_mlx/r2_map_apfs_workspace.py`; creation/mounting remains
  intentionally unimplemented while the daemon is hot.
- Blast radius: W4 source/static/protocol validation is unaffected, as are all
  datasets and checkpoints. The full heterogeneous correctness smoke and real
  model resource acceptance remain `pending-host-recovery`. The deterministic
  protocol fixture is evidence only and cannot substitute for those panels.
  Full evidence and recovery gates are in
  `docs/v2/reports/r2-map-w4-external-macho-gatekeeper-incident-v1.md`.
