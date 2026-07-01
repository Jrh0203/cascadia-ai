# ADR 0139: R0 Work-Conserving Host Rebalance

Status: accepted

Date: 2026-06-16

Experiment: `r0-spatial-footprint-screen-v1`

## Context

The corrected source-frozen R0 graph initially assigned train part 1 and
validation part 1 to john2. That host was already running the authorized
200-epoch local-geometry dropout origin. The other six R0 collection parts
finished in roughly one to two seconds each, leaving john1, john3, and john4
idle while every benchmark waited for the two unstarted john2 parts.

Collection host identity is not a scientific variable. Game-index interval,
rules, strategy, immutable source tree, executable, split, and final dataset
bytes are the controlled variables. Keeping the original host assignment would
add critical-path delay without adding evidence.

## Decision

`tools/r0_spatial_rebalance.py` atomically:

1. cancels the four unstarted john2 collection and fanout tasks with actor,
   reason, timestamp, and previous status;
2. creates a train-part-1 replacement on john3 and a validation-part-1
   replacement on john4;
3. preserves the exact game intervals, output roots, frozen source bundle, and
   collection arguments;
4. creates coordinator-owned whole-tree fanouts for both replacements; and
5. rewires all twelve benchmark tasks from the superseded fanout IDs to the
   replacement fanout IDs before releasing the queue lock.

The mutation fails atomically if a replacement ID exists, an original task is
active or complete, a benchmark task has started, or the source bundle cannot
satisfy the complete `cascadia-provenance` authority.

## Consequences

The 60,000-row corpus can close immediately on the free hosts. Once both
fanouts verify, john1, john3, and john4 can run their nine independent
benchmark processes while john2 continues MLX training. john2's three
partition-1 benchmark processes remain queued and start when the dropout
origin releases that host.

This changes scheduling only. It does not change accepted rows, source or
binary identity, modulo shard ownership, replicate count, model inputs, timing
protocol, or any R0 promotion gate.
