# Campaign State

Updated 2026-07-27.

## Resume here

The audit-heavy campaign system is retired. Do not resume old HOLD files,
waiters, receipt collectors, preregistered gates, source-identity repairs, or
seed-ledger chains.

Live state must be established from the machines themselves:

```bash
ps aux | grep -E 'cascadia|gumbel|torch|wildlife' | grep -v grep
ssh john0 "ps aux | grep -E 'cascadia|gumbel|torch' | grep -v grep; nvidia-smi"
for host in john1 john2 john3 john4; do
  ssh "$host" "ps aux | grep -E 'wildlife|cascadia' | grep -v grep" &
done
wait
```

At the time of this reset, no old process is presumed live merely because a
PID, heartbeat, done marker, or receipt exists. Verify the process.

## Player-strength campaign

- Goal: mean seat score at least 100 over 1,000 four-player self-play games.
- Historical best reference: approximately 98.4 at n1024/d16 under the
  corrected-rules campaign.
- Current approach: ordinary self-play generation, training, checkpointing,
  visible evaluation, and iterative tuning.
- Checkpoints need only be loadable and contain the model/training state needed
  to resume. Metrics JSONL and a short run config are sufficient bookkeeping.
- Any host may generate data, train, benchmark, or evaluate. Use concurrency
  that improves end-to-end throughput.

## Wildlife catalog

- 1,024/1,024 rulesets have a best-known board.
- 80/1,024 were exactly closed by the historical solver.
- Best known score: 85 on eight rulesets.
- Sound global interval: [85, 96].
- The two former score-97 branches, AADDB and ABDDB at counts
  `(4,0,6,4,6)`, were both proved infeasible at 85 by the July 26 connected
  runs. Their disconnected relaxations independently reached the same result.
- Current merged catalog:
  `docs/v3/evidence/all_wildlife_catalog_lean_top97_2026-07-26.json`.
- Continue from the existing catalog using the fastest solver available.
  Source hashes, taskset hashes, fleet receipts, and inherited-provenance
  checks are not prerequisites.
- Partial and heuristic improvements are welcome immediately. Mark a board
  `optimal` only when the exact solver actually finishes the proof; this is a
  mathematical distinction, not a provenance policy.

### Live fixed-count search

The fleet now runs the complete fixed-count candidate pipeline described in
[ALL_WILDLIFE_FIXED_COUNT_CATALOG.md](ALL_WILDLIFE_FIXED_COUNT_CATALOG.md).
It covers all 845,824 ruleset/count cells in atomic 256-cell chunks:

- shallow coverage: 8 × 20,000 iterations per cell, currently running;
- production depth: 12 × 100,000 iterations per cell, automatically queued
  behind shallow coverage on every host;
- john1-john4: shard indices 0-3 of four, eight search threads each.

The first pass stopped at 35.5% after its score-consistency assertion exposed
an order-dependent Elk-D recurrence. The corrected scorer explicitly optimizes
disjoint subsets around ring centers and uses a constant-size fast path for
the catalog's maximum-six constraint. Corrected output is schema v2. The 929
unaffected v1 chunks remain reusable; 244 already-generated v1 chunks
containing Elk D are excluded from progress and atomically regenerated.

At 02:09 EDT on July 27, corrected reusable coverage was 929 chunks / 237,824
boards (28.12%). All four hosts resumed at 02:11 EDT. Initial corrected chunks
took 26-42 seconds, projecting roughly 5-7 fleet-hours for the rest of shallow
coverage. The production stage remains approximately 36-55 additional hours.

The live pipeline configuration is
`cascadiav3/fleet/all_wildlife_fixed_count_pipeline_20260726.json`. john1 runs
in tmux session `cascadia-fixed-count-pipeline-john1`; the incremental central
sync runs in `cascadia-fixed-count-pipeline-sync`. Current activity can be
checked with:

```bash
tmux list-panes -t cascadia-fixed-count-pipeline-john1 \
  -F '#{pane_pid} #{pane_current_command}'
cat \
  cascadiav3/logs/all_wildlife_fixed_all_wildlife_fixed_count_shallow_20260726_john1.heartbeat
for host in john2 john3 john4; do
  ssh "$host" \
    "cat cascadia/cascadiav3/logs/all_wildlife_fixed_all_wildlife_fixed_count_shallow_20260726_${host}.heartbeat"
done
```

An hourly watchdog checks exact pipeline PIDs, twenty-minute stage-heartbeat
freshness, cross-hour chunk advancement, per-host disk headroom, and the
central synchronizer. It automatically resumes absent or
forty-five-minute-progress-stalled hosts from their atomic chunks and appends snapshots to
`cascadiav3/logs/all_wildlife_fixed_count_watchdog_20260726.jsonl`. The
headless john1 session uses the installer's idempotent hourly crontab fallback.
Central recovery continues after worker completion until the deterministic
merged catalog has independently rescored every source and materialized output
board.

## Next actions

The audit/provenance teardown is complete across the active runners, solvers,
training readers, cluster helpers, and documentation. Legacy gate tools were
removed rather than kept as dormant complexity. No live computation was
stopped or replaced during the teardown.

Let the fixed-count pipeline and hourly watchdog run. Inspect the atomic
summary at
`cascadiav3/fleet_outputs/all_wildlife_fixed_count_shallow_20260726/summary.json`.
It is safe to stop and resume at any point; only one unpublished chunk per host
can be lost. Deeply validate the completed shallow stage, then merge each
production chunk by maximum score while retaining the shallow board whenever
it remains stronger. The synchronized pipeline performs that deterministic
merge automatically into
`cascadiav3/fleet_outputs/all_wildlife_fixed_count_best_20260726/`.
