#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
TASK_INDICES="${TASK_INDICES:?set TASK_INDICES}"
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

cd "$ROOT"
mkdir -p "$LOG_DIR"
test -x "$WORKER"
mkdir -p "$OUTPUT_DIR"
rm -f "$PID_FILE"

/usr/bin/nohup /bin/bash "$WORKER" > "$LOG" 2>&1 < /dev/null &
nohup_pid=$!

printf '%s\n' "$nohup_pid" > "$PID_FILE"
printf '%s\n' "$nohup_pid"
