#!/usr/bin/env bash
# Supervisor wrapper: relaunches run_overnight_phase2.sh on SIGPIPE (141)
# or any non-zero / non-halt-gate exit. The inner runner is crash-resumable
# via state.json so each restart continues at the next un-completed iter.

LOG="alphazero_v2_run/supervisor.log"
RUNNER="./run_overnight_phase2.sh"

mkdir -p "$(dirname "$LOG")"

for attempt in $(seq 1 200); do
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] supervisor: attempt $attempt — launching $RUNNER" >> "$LOG"
    "$RUNNER"
    rc=$?
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] supervisor: attempt $attempt exited rc=$rc" >> "$LOG"
    # rc=0  → runner completed all iters (normal finish)
    # rc=2  → runner aborted via halt-gate (don't restart, diverging)
    # other → SIGPIPE / crash → restart after 5s
    if [[ $rc -eq 0 ]]; then
        echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] supervisor: runner finished cleanly, done" >> "$LOG"
        exit 0
    fi
    if [[ $rc -eq 2 ]]; then
        echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] supervisor: runner halt-gated, not restarting" >> "$LOG"
        exit 2
    fi
    sleep 5
done
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] supervisor: 200 attempts exhausted, giving up" >> "$LOG"
exit 1
