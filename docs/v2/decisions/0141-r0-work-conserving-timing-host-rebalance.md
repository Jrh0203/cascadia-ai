# ADR 0141: R0 Work-Conserving Timing Host Rebalance

Status: accepted

Date: 2026-06-16

Experiment: `r0-spatial-footprint-screen-v1`

## Context

The corrected R0 corpus, immutable executable bundle, and clean timing wave
produced nine of twelve accepted process reports:

- shard 0, replicas 0 through 2 on john1;
- shard 2, replicas 0 through 2 on john3; and
- shard 3, replicas 0 through 2 on john4.

Shard 1 remained unstarted on john2 because the already-authorized
`conditional-tile-local-geometry-dropout-v1` MLX origin exclusively occupied
that host. john3 and john4 were healthy and idle, and every source-frozen
dataset tree plus the exact benchmark bundle was already byte-identical on
both hosts.

Waiting for john2 would add critical-path wall time without adding scientific
control. R0 compares all five arms inside each process and reports paired
within-process ratios. Host identity is recorded provenance, not a treatment.
ADR 0136 requires the three replicas for one shard to have identical
non-timing identity, including one common execution host; it does not require
different shards to occupy different hosts.

## Decision

Move the complete unstarted shard-1 replicate group to john3:

```text
shard 1, replicate 0 -> john3
shard 1, replicate 1 -> john3
shard 1, replicate 2 -> john3
```

All three replacements:

- use the immutable bundle
  `c4e99c53462e9884c0d9bbbb2220fb70429ae71f6486c7769441e85f1a5750d9`;
- consume the same eight ordered source-frozen dataset manifests;
- retain `global_ordinal % 4 == 1`;
- run all five arms in one release process;
- use 50 benchmark iterations;
- write new immutable report paths; and
- execute sequentially on one host, preserving independent process
  replication and timing-free replica identity.

The three original john2 tasks are administratively cancelled before they
start. The existing clean collection, forward classifier, reverse classifier,
and merge-order proof are also cancelled because their frozen paths name the
superseded john2 reports. New downstream tasks consume only:

- the three accepted clean john1 reports;
- the three reassigned john3 shard-1 reports;
- the three accepted john3 shard-2 reports; and
- the three accepted john4 shard-3 reports.

No completed task or artifact is mutated or deleted.

## Atomicity

`tools/r0_timing_host_rebalance.py` installs the replacement graph under one
queue lock. It fails before mutation unless:

- all nine already accepted reports have completed queue tasks;
- every superseded task is present, unclaimed, and still pending or failed;
- no replacement task ID already exists;
- the immutable bundle passes whole-tree validation; and
- the replacement host is john3 or john4.

The tool records actor, reason, prior status, and timestamp for every
administrative cancellation.

## Scientific Validity

The change does not alter:

- source or executable identity;
- dataset identity, order, row count, or split;
- shard membership;
- arm definitions;
- semantic round-trip requirements;
- iteration count;
- within-process pairing;
- process-replica count;
- median-process selection;
- classifier logic; or
- forward/reverse byte-identity proof.

It changes only the nuisance execution host for one complete shard group.
Because all three shard-1 replicas move together, ADR 0136's replica-identity
gate remains exact.

## Consequences

R0 can finish while dropout training continues on john2. john3 performs two
different nonoverlapping shard groups sequentially; this is additional unique
evidence, not duplicate work. Cross-host absolute timing remains descriptive,
while the preregistered decision evidence continues to use paired
within-process ratios.

The original john2 tasks remain visible as cancelled audit history. If any had
started before the atomic migration, this ADR's tool would reject the change
and require a new recovery decision.
