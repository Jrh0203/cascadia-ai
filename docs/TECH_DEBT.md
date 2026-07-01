# Technical Debt

New debt or unavoidable compromises must be documented here with cause, proper
fix, and blast radius before merge.

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
