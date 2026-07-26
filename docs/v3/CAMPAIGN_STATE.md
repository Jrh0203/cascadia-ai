# Campaign State

Updated 2026-07-26.

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

### Live exact search

A 32-case CP-SAT queue started on 2026-07-26 at approximately 12:10 EDT and
is running across john1-john4. Each host has eight sequential cases, eight
solver workers per case, a 5h45m solve limit, and a 5h50m total case limit.
The worst-case queue length is 46h40m, ending by approximately 10:50 EDT on
2026-07-28.

The queue contains all 19 unresolved count branches whose current upper bound
is 96, followed by 13 branches at 95. Therefore the first five queue positions
on john1-john3 and the first four on john4 cover the complete score-96
frontier. If every one is either excluded or tightened below 96, the global
upper bound falls to 95.

The frozen taskset is
`cascadiav3/fleet/lean_bound_48h_20260726_taskset.json`. Results are written as
`task_*.json` under
`cascadiav3/fleet_outputs/lean_bound_48h_20260726/` on each host. The host
assignments are:

- john1: `0,4,8,12,16,20,24,28`;
- john2: `1,5,9,13,17,21,25,29`;
- john3: `2,6,10,14,18,22,26,30`;
- john4: `3,7,11,15,19,23,27,31`.

As of 12:10 EDT, all four solvers were healthy at approximately 750-800% CPU.
john1 runs in the tmux session `cascadia-lean-bound-48h-john1`; john2-john4
run under their detached host wrappers. Current activity can be checked with:

```bash
tmux list-panes -t cascadia-lean-bound-48h-john1 \
  -F '#{pane_pid} #{pane_current_command}'
cat cascadiav3/logs/all_wildlife_bound_lean_bound_48h_20260726_john1.heartbeat
for host in john2 john3 john4; do
  ssh "$host" \
    "cat cascadia/cascadiav3/logs/all_wildlife_bound_lean_bound_48h_20260726_${host}.heartbeat"
done
```

## Next actions

The audit/provenance teardown is complete across the active runners, solvers,
training readers, cluster helpers, and documentation. Legacy gate tools were
removed rather than kept as dormant complexity. No live computation was
stopped or replaced during the teardown.

Let the four queues run. Completed remote files can be copied without stopping
the solvers and merged at any time:

```bash
for host in john2 john3 john4; do
  rsync -a "$host:cascadia/cascadiav3/fleet_outputs/lean_bound_48h_20260726/" \
    cascadiav3/fleet_outputs/lean_bound_48h_20260726/
done
.venv/bin/python -m tools.all_wildlife_bound_probe_collect \
  --base-catalog \
    docs/v3/evidence/all_wildlife_catalog_lean_top97_2026-07-26.json \
  --probe-directories cascadiav3/fleet_outputs/lean_bound_48h_20260726 \
  --output docs/v3/evidence/all_wildlife_catalog_lean_48h_2026-07-28.json \
  --markdown docs/v3/ALL_WILDLIFE_CATALOG_LEAN_48H.md
```

A better witness raises the stored board; a lower objective bound contracts
the holistic interval. Do not interrupt the queues to recreate any retired
campaign infrastructure.
