#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
TASK_INDICES="${TASK_INDICES:?set TASK_INDICES}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
TASKSET_SHA256="${TASKSET_SHA256:?set TASKSET_SHA256}"
CATALOG_SHA256="${CATALOG_SHA256:?set CATALOG_SHA256}"
PROBE_SOURCE_SHA256="${PROBE_SOURCE_SHA256:?set PROBE_SOURCE_SHA256}"
EXACT_SOURCE_SHA256="${EXACT_SOURCE_SHA256:?set EXACT_SOURCE_SHA256}"
EXACT_SUPPORT_SHA256="${EXACT_SUPPORT_SHA256:?set EXACT_SUPPORT_SHA256}"
RULES_SOURCE_SHA256="${RULES_SOURCE_SHA256:?set RULES_SOURCE_SHA256}"
WILDLIFE_VENV="${WILDLIFE_VENV:?set WILDLIFE_VENV}"
TIME_LIMIT="${TIME_LIMIT:-300}"
TOTAL_TIME_LIMIT="${TOTAL_TIME_LIMIT:-330}"
SOLVER_WORKERS="${SOLVER_WORKERS:-8}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-5}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$TASK_INDICES:$TIME_LIMIT:$TOTAL_TIME_LIMIT:$SOLVER_WORKERS:$HEARTBEAT_INTERVAL" in
  *[!0-9,.:]*)
    echo "task indices and solver settings contain invalid characters" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
LOG_DIR="${ROOT}/cascadiav3/logs"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
WORKER="${ROOT}/cascadiav3/scripts/fleet_all_wildlife_bound_probe_worker.sh"
LOG="${LOG_DIR}/all_wildlife_bound_${FLEET_TAG}_${SHARD_HOST}.log"
PID_FILE="${LOG_DIR}/all_wildlife_bound_${FLEET_TAG}_${SHARD_HOST}.pid"
EXIT_FILE="${LOG_DIR}/all_wildlife_bound_${FLEET_TAG}_${SHARD_HOST}.exit"
HEARTBEAT="${LOG_DIR}/all_wildlife_bound_${FLEET_TAG}_${SHARD_HOST}.heartbeat"

cd "$ROOT"
mkdir -p "$LOG_DIR"
test -x "$WORKER"
test ! -e "$OUTPUT_DIR"
test ! -e "$PID_FILE"
test ! -e "$EXIT_FILE"
test ! -e "$HEARTBEAT"
test ! -e "$LOG"

/usr/bin/nohup /bin/bash "$WORKER" > "$LOG" 2>&1 < /dev/null &
nohup_pid=$!

# A launch is successful only when the worker has written its own durable PID
# and heartbeat and that exact PID is live. The nohup child PID is diagnostic;
# the worker-owned PID is authoritative.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if [ -s "$PID_FILE" ] && [ -s "$HEARTBEAT" ]; then
    break
  fi
  sleep 1
done
test -s "$PID_FILE"
test -s "$HEARTBEAT"
worker_pid="$(tr -d '\n' < "$PID_FILE")"
case "$worker_pid" in
  ""|*[!0-9]*)
    echo "worker wrote invalid PID: $worker_pid" >&2
    exit 70
    ;;
esac
kill -0 "$worker_pid"
if [ "$worker_pid" != "$nohup_pid" ]; then
  echo "worker PID $worker_pid differs from nohup child $nohup_pid" >&2
  exit 70
fi
printf '%s\n' "$worker_pid"
