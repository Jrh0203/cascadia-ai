#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
SHARD_INDEX="${SHARD_INDEX:?set zero-based SHARD_INDEX}"
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
    echo "fixed-count worker settings must be nonnegative integers" >&2
    exit 64
    ;;
esac
if [ "$SHARD_COUNT" -lt 1 ] || [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
  echo "SHARD_INDEX must be smaller than positive SHARD_COUNT" >&2
  exit 64
fi
for positive in \
  "$CHUNK_SIZE" "$TOTAL_CELLS" "$SEARCH_THREADS" "$RESTARTS_PER_CELL" \
  "$ITERATIONS_PER_RESTART" "$HEARTBEAT_INTERVAL"; do
  if [ "$positive" -lt 1 ]; then
    echo "worker sizes and search settings must be positive" >&2
    exit 64
  fi
done
if [ "$HEARTBEAT_INTERVAL" -gt 60 ]; then
  echo "HEARTBEAT_INTERVAL must not exceed 60 seconds" >&2
  exit 64
fi

ROOT="${HOME}/cascadia"
BINARY="${ROOT}/target/release/all_wildlife_candidates"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"
HEARTBEAT="${LOG_DIR}/all_wildlife_fixed_${FLEET_TAG}_${SHARD_HOST}.heartbeat"
EXIT_FILE="${LOG_DIR}/all_wildlife_fixed_${FLEET_TAG}_${SHARD_HOST}.exit"
CHILD_PID_FILE="${LOG_DIR}/all_wildlife_fixed_${FLEET_TAG}_${SHARD_HOST}.solver.pid"
WRAPPER_PID_FILE="${LOG_DIR}/all_wildlife_fixed_${FLEET_TAG}_${SHARD_HOST}.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$BINARY"
rm -f "$EXIT_FILE"
printf '%s\n' "$$" > "${WRAPPER_PID_FILE}.tmp"
mv "${WRAPPER_PID_FILE}.tmp" "$WRAPPER_PID_FILE"

solver_pid=""
heartbeat_pid=""
terminate_worker() {
  trap - TERM INT HUP
  if [ -n "$heartbeat_pid" ] && kill -0 "$heartbeat_pid" 2>/dev/null; then
    kill -TERM "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  if [ -n "$solver_pid" ] && kill -0 "$solver_pid" 2>/dev/null; then
    kill -TERM "$solver_pid" 2>/dev/null || true
    wait "$solver_pid" 2>/dev/null || true
  fi
  printf '143\n' > "${EXIT_FILE}.tmp"
  mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
  exit 143
}
trap terminate_worker TERM INT HUP

total_chunks=$(((TOTAL_CELLS + CHUNK_SIZE - 1) / CHUNK_SIZE))
for ((chunk_index = SHARD_INDEX; chunk_index < total_chunks; chunk_index += SHARD_COUNT)); do
  range_start=$((chunk_index * CHUNK_SIZE))
  range_end=$((range_start + CHUNK_SIZE))
  if [ "$range_end" -gt "$TOTAL_CELLS" ]; then
    range_end="$TOTAL_CELLS"
  fi
  output="$(printf '%s/chunk_%05d.json' "$OUTPUT_DIR" "$chunk_index")"
  printf '%s host=%s chunk=%s cells=%s..%s output=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SHARD_HOST" "$chunk_index" \
    "$range_start" "$range_end" "$output"

  "$BINARY" fixed-count-chunk "$output" "$range_start" "$range_end" \
    "$SEARCH_THREADS" "$RESTARTS_PER_CELL" "$ITERATIONS_PER_RESTART" \
    "$BASE_SEED" &
  solver_pid=$!
  printf '%s\n' "$solver_pid" > "${CHILD_PID_FILE}.tmp"
  mv "${CHILD_PID_FILE}.tmp" "$CHILD_PID_FILE"
  (
    while kill -0 "$solver_pid" 2>/dev/null; do
      printf '%s solver_pid=%s chunk=%s cells=%s..%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" "$chunk_index" \
        "$range_start" "$range_end" > "${HEARTBEAT}.tmp"
      mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
      sleep "$HEARTBEAT_INTERVAL"
    done
  ) &
  heartbeat_pid=$!
  set +e
  wait "$solver_pid"
  status=$?
  set -e
  solver_pid=""
  kill -TERM "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  heartbeat_pid=""
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "$status" > "${EXIT_FILE}.tmp"
    mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
    exit "$status"
  fi
done

printf '%s complete chunks=%s shard=%s/%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$total_chunks" "$SHARD_INDEX" \
  "$SHARD_COUNT" > "${HEARTBEAT}.tmp"
mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
printf '0\n' > "${EXIT_FILE}.tmp"
mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
