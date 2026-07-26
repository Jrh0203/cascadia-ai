#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX}"
SHARD_COUNT="${SHARD_COUNT:?set SHARD_COUNT}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
TOTAL_CELLS="${TOTAL_CELLS:-845824}"
SEARCH_THREADS="${SEARCH_THREADS:-8}"
RESTARTS_PER_CELL="${RESTARTS_PER_CELL:-8}"
ITERATIONS_PER_RESTART="${ITERATIONS_PER_RESTART:-20000}"
BASE_SEED="${BASE_SEED:-20260726}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-15}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$SHARD_INDEX:$SHARD_COUNT:$CHUNK_SIZE:$TOTAL_CELLS:$SEARCH_THREADS:$RESTARTS_PER_CELL:$ITERATIONS_PER_RESTART:$BASE_SEED:$HEARTBEAT_INTERVAL" in
  *[!0-9:]*)
    echo "fixed-count launcher settings must be nonnegative integers" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
LOG_DIR="${ROOT}/cascadiav3/logs"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
WORKER="${ROOT}/cascadiav3/scripts/fleet_all_wildlife_fixed_count_candidates_worker.sh"
LOG="${LOG_DIR}/all_wildlife_fixed_${FLEET_TAG}_${SHARD_HOST}.log"
PID_FILE="${LOG_DIR}/all_wildlife_fixed_${FLEET_TAG}_${SHARD_HOST}.pid"

cd "$ROOT"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
test -x "$WORKER"
rm -f "$PID_FILE"

/usr/bin/nohup /bin/bash "$WORKER" > "$LOG" 2>&1 < /dev/null &
nohup_pid=$!
printf '%s\n' "$nohup_pid" > "$PID_FILE"
printf '%s\n' "$nohup_pid"
