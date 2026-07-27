# All-ruleset fixed-count wildlife catalog

This run generates a strong connected 20-animal board for every pair of:

- the 1,024 ordered Bear/Elk/Salmon/Hawk/Fox A-D card sets; and
- the 826 legal five-species count vectors summing to 20 with no count above
  six.

The dense catalog therefore contains 845,824 cells. A cell is a
`(ruleset, counts)` optimization problem, not an exact-optimality claim.

## Search schedule

The four Mac minis divide 3,304 chunks round-robin. Each chunk contains 256
cells and uses eight search threads.

1. The shallow stage runs 8 restarts × 20,000 annealing iterations per cell.
   It gives complete coverage quickly.
2. The production stage starts automatically after the shallow stage on each
   host and runs 12 restarts × 100,000 iterations per cell.
3. The best board across both stages is retained during final collection.
   Exact CP-SAT work remains selective because proving every cell would be
   dominated by the extreme timeout tail.

After both stages are synchronized, the pipeline automatically writes the
deterministic best board for every cell under:

```text
cascadiav3/fleet_outputs/all_wildlife_fixed_count_best_20260726/
```

Production wins only when its score is higher. Ties use canonical token JSON
and then stage order, making repeated merges byte-stable.

The run configuration is
`cascadiav3/fleet/all_wildlife_fixed_count_pipeline_20260726.json`.

## Incremental durability

Every `chunk_XXXXX.json` is written to a temporary path, completely generated,
and validated in Rust before one atomic rename publishes it. Validation checks:

- the flattened cell, ruleset, and count-vector identities;
- exactly 20 nonoverlapping connected coordinates;
- the requested animal counts;
- the stored score and five-part production score; and
- the sound count upper stored by the generator.

On restart, every existing chunk assigned to that host is loaded, deeply
validated, and skipped. An interruption can lose only the active unpublished
chunk on each host—normally tens of seconds in the shallow stage. Completed
chunks never need to be regenerated.

### Elk-D scoring revision

The first run exposed an order-dependence bug in Elk-D scoring: the old
set-packing recurrence greedily claimed every still-unassigned elk around a
ring center. Reordering the identical coordinates could therefore change the
score. The corrected scorer explicitly considers every nonempty subset around
each center and maximizes over disjoint groups. A constant-size bitmask fast
path keeps the common zero-to-six-elk case inexpensive.

Corrected chunks use schema
`all-wildlife-fixed-count-candidates-v2`. On resume, v1 chunks that do not
contain Elk D are still deeply validated and reused. V1 chunks containing Elk
D are regenerated and replaced atomically only after the corrected chunk is
complete. The collector excludes those stale v1 chunks from its completion
count until replacement, so reported progress always means reusable,
current-scoring work.

john1 synchronizes finished chunks from john2-john4 every five minutes and
writes an atomic partial summary. Synchronization accepts a remote chunk only
when it is newer than the central copy, allowing corrected v2 replacements
without permitting an older v1 file to overwrite them. The shallow summary is:

```text
cascadiav3/fleet_outputs/all_wildlife_fixed_count_shallow_20260726/summary.json
```

The production summary will use the analogous
`all_wildlife_fixed_count_production_20260726` directory.

## Safe interruption and resume

Terminate the pipeline wrapper, not only the active Rust child. The wrapper
forwards the signal through the stage worker to the active search:

```bash
tmux kill-session -t cascadia-fixed-count-pipeline-john1
for host in john2 john3 john4; do
  ssh "$host" '
    tag=all_wildlife_fixed_count_pipeline_20260726
    kill -TERM "$(cat "cascadia/cascadiav3/logs/all_wildlife_fixed_pipeline_${tag}_$(hostname -s).pid")"
  '
done
```

Relaunching the same pipeline configuration resumes from the chunk files. Do
not change the seed or per-stage search budget under an existing stage tag:
the validator intentionally rejects that configuration mismatch.

## Partial validation

Any synchronized prefix or subset can be checked without stopping computation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  -m tools.all_wildlife_fixed_count_collect \
  --directories \
    cascadiav3/fleet_outputs/all_wildlife_fixed_count_shallow_20260726 \
  --chunk-size 256 \
  --deep \
  --output \
    cascadiav3/fleet_outputs/all_wildlife_fixed_count_shallow_20260726/summary.json
```

`--deep` independently recomputes every species score in Python. The recurring
five-minute synchronizer performs the cheaper structural validation; the final
collection performs the deep pass.

## Hourly health monitoring

The watchdog checks all four pipeline PIDs and stage heartbeats, verifies the
central synchronizer, records a JSONL health snapshot, and restarts an absent
or twenty-minute-stale pipeline from its existing chunks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  tools/all_wildlife_fixed_count_watchdog.py
```

The macOS installer registers that check as a per-user background launchd job
with a one-hour interval. In a headless/background login domain where launchd
rejects a LaunchAgent, it installs the equivalent idempotent hourly crontab
entry instead:

```bash
cascadiav3/scripts/install_all_wildlife_fixed_count_watchdog.sh
```

Health history is appended to
`cascadiav3/logs/all_wildlife_fixed_count_watchdog_20260726.jsonl`.
