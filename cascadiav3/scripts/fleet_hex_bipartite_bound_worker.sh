#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
START_INDEX="${START_INDEX:?set START_INDEX}"
END_INDEX="${END_INDEX:?set END_INDEX}"
SOURCE_SHA256="${SOURCE_SHA256:?set SOURCE_SHA256}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"
SECONDS_PER_COMPONENT="${SECONDS_PER_COMPONENT:-120}"
SOLVER_WORKERS="${SOLVER_WORKERS:-8}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$START_INDEX:$END_INDEX:$SECONDS_PER_COMPONENT:$SOLVER_WORKERS" in
  *[!0-9.:]*)
    echo "numeric worker arguments contain invalid characters" >&2
    exit 64
    ;;
esac
case "$WILDLIFE_VENV" in
  ""|/*|*".."*|*[!A-Za-z0-9._/-]*)
    echo "WILDLIFE_VENV must be a safe relative path" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
PYTHON="${ROOT}/${WILDLIFE_VENV}/bin/python"
SOURCE="${ROOT}/tools/derive_hex_bipartite_edge_bounds.py"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"
OUTPUT="${OUTPUT_DIR}/shard_${SHARD_HOST}.json"
HEARTBEAT="${LOG_DIR}/hex_bound_${FLEET_TAG}_${SHARD_HOST}.heartbeat"
EXIT_FILE="${LOG_DIR}/hex_bound_${FLEET_TAG}_${SHARD_HOST}.exit"
CHILD_PID_FILE="${LOG_DIR}/hex_bound_${FLEET_TAG}_${SHARD_HOST}.solver.pid"
WRAPPER_PID_FILE="${LOG_DIR}/hex_bound_${FLEET_TAG}_${SHARD_HOST}.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"
test ! -e "$OUTPUT"
test ! -e "$EXIT_FILE"
test ! -e "$WRAPPER_PID_FILE"
observed_source="$(shasum -a 256 "$SOURCE" | awk '{print $1}')"
[ "$observed_source" = "$SOURCE_SHA256" ]
printf '%s\n' "$$" > "${WRAPPER_PID_FILE}.tmp"
mv "${WRAPPER_PID_FILE}.tmp" "$WRAPPER_PID_FILE"

set +e
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u -m tools.derive_hex_bipartite_edge_bounds \
  --start-index "$START_INDEX" --end-index "$END_INDEX" \
  --seconds "$SECONDS_PER_COMPONENT" --workers "$SOLVER_WORKERS" \
  --output "$OUTPUT" &
solver_pid=$!
printf '%s\n' "$solver_pid" > "${CHILD_PID_FILE}.tmp"
mv "${CHILD_PID_FILE}.tmp" "$CHILD_PID_FILE"
while kill -0 "$solver_pid" 2>/dev/null; do
  printf '%s solver_pid=%s range=[%s,%s)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" "$START_INDEX" "$END_INDEX" \
    > "${HEARTBEAT}.tmp"
  mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
  sleep 30
done
wait "$solver_pid"
worker_status=$?
set -e

printf '%s solver_pid=%s exit=%s range=[%s,%s)\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" "$worker_status" \
  "$START_INDEX" "$END_INDEX" > "${HEARTBEAT}.tmp"
mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
printf '%s\n' "$worker_status" > "${EXIT_FILE}.tmp"
mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
exit "$worker_status"
