# R2-MAP W6 deterministic-controller readmission v1 preregistration

Date: 2026-06-18

Status: frozen isolated validation protocol; canonical initialization, phase
transition, W7 launch, and protected-domain access are not authorized

## Question

Does the current controller deterministically traverse every bootstrap,
rejection, promotion, retry, recovery, queue, ledger, receipt, and dashboard
projection shape without mutating the canonical revision-0 campaign?

## Source and storage boundary

Run only from the same immutable expanded John2 source transaction accepted by
W0 v1.1 and W5. The execution root must be a fresh child of the remote-run
`TMPDIR` beneath `/Users/john2/cascadia-bench/r2-map-v1/tmp/`; it must not be
the canonical campaign root. All durable source, run, report, receipt, and
cleanup evidence remains on John2's internal disk. The SSD and John4 are
forbidden.

## Frozen validation

1. Rebuild and verify the deterministic source manifest before execution.
2. Snapshot the canonical campaign state, decision log, queue, ledger,
   controller history/stop, work-packet and incoming-receipt directories,
   controller schemas, controller dashboard inputs, and terminal/stop objects.
3. Run the complete Python controller suite and Ruff checks for controller,
   contracts, dashboard projection, queue, ledger, and CLI sources.
4. Execute `tools/r2_map_expert_iteration.py w6-dry-run` against exactly
   `$TMPDIR/w6-isolated-controller-v1`.
5. Require 21 hash-chained transitions, both rejection and promotion branches,
   final phase `incumbent-promoted`, promotion index 1, round index 1, exactly
   30 completed queue tasks, 30 work receipts, 30 synthetic storage receipts,
   60 total receipts, one ledger projection, and no stop file.
6. Hash every regular isolated-tree file, reject links/special files, retain
   the embedded dry-run report and tree identity in the validation summary,
   and delete the isolated and pytest trees.
7. Re-snapshot canonical control state and require byte-identical equality.

The outer receipt must bind exact argv, exit status, stdout/stderr hashes,
source identity, and run cleanup. Independent John2 and John3 verifiers reopen
the source, run, cleanup, publication, and report receipts.

## Failure rule

Any command failure, source drift, canonical-state change, unexpected tree
entry, wrong count, stop/terminal creation, cleanup residue, or missing receipt
keeps W6 red. An isolated pass does not authorize canonical initialization.
Canonical queue/ledger/schema/dashboard-input initialization remains a later,
explicit storage-owner action after the aggregate W0-W6 readmission packet is
accepted.
