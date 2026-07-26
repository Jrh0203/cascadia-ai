#!/usr/bin/env bash
set -euo pipefail

PIPELINE_TAG="${PIPELINE_TAG:?set PIPELINE_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX}"
SHARD_COUNT="${SHARD_COUNT:?set SHARD_COUNT}"
SHALLOW_TAG="${SHALLOW_TAG:?set SHALLOW_TAG}"
PRODUCTION_TAG="${PRODUCTION_TAG:?set PRODUCTION_TAG}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
TOTAL_CELLS="${TOTAL_CELLS:-845824}"
SEARCH_THREADS="${SEARCH_THREADS:-8}"
BASE_SEED="${BASE_SEED:-20260726}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-15}"
SHALLOW_RESTARTS="${SHALLOW_RESTARTS:-8}"
SHALLOW_ITERATIONS="${SHALLOW_ITERATIONS:-20000}"
PRODUCTION_RESTARTS="${PRODUCTION_RESTARTS:-12}"
PRODUCTION_ITERATIONS="${PRODUCTION_ITERATIONS:-100000}"

case "$PIPELINE_TAG:$SHARD_HOST:$SHALLOW_TAG:$PRODUCTION_TAG" in
  *[!A-Za-z0-9._:-]*)
    echo "pipeline tags and host must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$SHARD_INDEX:$SHARD_COUNT:$CHUNK_SIZE:$TOTAL_CELLS:$SEARCH_THREADS:$BASE_SEED:$HEARTBEAT_INTERVAL:$SHALLOW_RESTARTS:$SHALLOW_ITERATIONS:$PRODUCTION_RESTARTS:$PRODUCTION_ITERATIONS" in
  *[!0-9:]*)
    echo "pipeline settings must be nonnegative integers" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
WORKER="${ROOT}/cascadiav3/scripts/fleet_all_wildlife_fixed_count_candidates_worker.sh"
LOG_DIR="${ROOT}/cascadiav3/logs"
PID_FILE="${LOG_DIR}/all_wildlife_fixed_pipeline_${PIPELINE_TAG}_${SHARD_HOST}.pid"
EXIT_FILE="${LOG_DIR}/all_wildlife_fixed_pipeline_${PIPELINE_TAG}_${SHARD_HOST}.exit"
STAGE_FILE="${LOG_DIR}/all_wildlife_fixed_pipeline_${PIPELINE_TAG}_${SHARD_HOST}.stage"

cd "$ROOT"
mkdir -p "$LOG_DIR"
test -x "$WORKER"
rm -f "$EXIT_FILE"
printf '%s\n' "$$" > "${PID_FILE}.tmp"
mv "${PID_FILE}.tmp" "$PID_FILE"

worker_pid=""
terminate_pipeline() {
  trap - TERM INT HUP
  if [ -n "$worker_pid" ] && kill -0 "$worker_pid" 2>/dev/null; then
    kill -TERM "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
  fi
  printf '143\n' > "${EXIT_FILE}.tmp"
  mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
  exit 143
}
trap terminate_pipeline TERM INT HUP

run_stage() {
  stage_tag="$1"
  restarts="$2"
  iterations="$3"
  printf '%s\n' "$stage_tag" > "${STAGE_FILE}.tmp"
  mv "${STAGE_FILE}.tmp" "$STAGE_FILE"
  printf '%s host=%s stage=%s restarts=%s iterations=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SHARD_HOST" "$stage_tag" \
    "$restarts" "$iterations"
  FLEET_TAG="$stage_tag" SHARD_HOST="$SHARD_HOST" \
    SHARD_INDEX="$SHARD_INDEX" SHARD_COUNT="$SHARD_COUNT" \
    CHUNK_SIZE="$CHUNK_SIZE" TOTAL_CELLS="$TOTAL_CELLS" \
    SEARCH_THREADS="$SEARCH_THREADS" RESTARTS_PER_CELL="$restarts" \
    ITERATIONS_PER_RESTART="$iterations" BASE_SEED="$BASE_SEED" \
    HEARTBEAT_INTERVAL="$HEARTBEAT_INTERVAL" \
    /bin/bash "$WORKER" &
  worker_pid=$!
  set +e
  wait "$worker_pid"
  status=$?
  set -e
  worker_pid=""
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "$status" > "${EXIT_FILE}.tmp"
    mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
    exit "$status"
  fi
}

run_stage "$SHALLOW_TAG" "$SHALLOW_RESTARTS" "$SHALLOW_ITERATIONS"
run_stage "$PRODUCTION_TAG" "$PRODUCTION_RESTARTS" "$PRODUCTION_ITERATIONS"

printf 'complete\n' > "${STAGE_FILE}.tmp"
mv "${STAGE_FILE}.tmp" "$STAGE_FILE"
printf '0\n' > "${EXIT_FILE}.tmp"
mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
