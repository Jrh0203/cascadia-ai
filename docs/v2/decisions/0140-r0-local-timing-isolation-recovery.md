# ADR 0140: R0 Local Timing Isolation Recovery

Status: accepted

Date: 2026-06-16

Experiment: `r0-spatial-footprint-screen-v1`

## Context

The first john1 R0 timing process began while two local F5 Rust verification
commands were still running. The source, records, round trips, and semantic
hashes were valid, but CPU contention made its timing measurements
inadmissible. The overlap continued into the next local process before it was
terminated. john3 and john4 had no overlapping coordinator-owned workload and
their timing processes remain admissible.

Overwriting the original report or treating the overlap as ordinary variance
would erase provenance. Reusing the completed task ID would also make the queue
claim evidence disagree with the artifact now stored at its path.

## Decision

`tools/r0_timing_recovery.py` installs a separate clean local wave:

- three new john1 benchmark task IDs and artifact paths;
- the same 60,000 rows, shard 0 modulo rule, source bundle, arm set, 50
  iterations, and replicate indices 0, 1, and 2;
- a replacement remote-report collection gate;
- clean forward and reverse classifiers that reference only the three new
  john1 reports and the nine admissible remote reports; and
- a replacement byte-level merge-order proof.

The original completed report remains immutable and quarantined. The
terminated process, unstarted third process, and original downstream
classification graph are administratively cancelled with an explicit reason.
No contaminated local report is accepted by the replacement classifier.

The recovery fails atomically unless the completed contaminated task, cancelled
unstarted task, failed terminated task, original downstream graph, and all
replacement IDs have exactly the expected states.

## Consequences

R0 keeps twelve independent accepted process reports without discarding the
remote work or concealing the local overlap. This adds three local processes,
roughly two minutes, and preserves the preregistered median-process and
replicate-variance analysis.
