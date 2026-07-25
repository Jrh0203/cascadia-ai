# Campaign State

Updated 2026-07-25.

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
- Historical global upper bound: 97.
- Continue from the existing catalog using the fastest solver available.
  Source hashes, taskset hashes, fleet receipts, and inherited-provenance
  checks are not prerequisites.
- Partial and heuristic improvements are welcome immediately. Mark a board
  `optimal` only when the exact solver actually finishes the proof; this is a
  mathematical distinction, not a provenance policy.

## Next actions

The audit/provenance teardown is complete across the active runners, solvers,
training readers, cluster helpers, and documentation. Legacy gate tools were
removed rather than kept as dormant complexity. No live computation was
stopped or replaced during the teardown.

Next, establish the actual live process/checkpoint state and resume whichever
strength or wildlife search has the highest expected value. Benchmark
concurrency in the course of useful work instead of creating a separate
approval campaign around it.
