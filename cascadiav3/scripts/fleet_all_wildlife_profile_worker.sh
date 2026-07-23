#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
INDICES="${INDICES:?set comma-separated INDICES}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
TASKSET_SHA256="${TASKSET_SHA256:?set TASKSET_SHA256}"
RULES_SOURCE_SHA256="${RULES_SOURCE_SHA256:?set RULES_SOURCE_SHA256}"
EXACT_SOURCE_SHA256="${EXACT_SOURCE_SHA256:?set EXACT_SOURCE_SHA256}"
EXACT_SUPPORT_SHA256="${EXACT_SUPPORT_SHA256:?set EXACT_SUPPORT_SHA256}"
RUNNER_SOURCE_SHA256="${RUNNER_SOURCE_SHA256:?set RUNNER_SOURCE_SHA256}"
WORKER_SHA256="${WORKER_SHA256:?set WORKER_SHA256}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"
SECONDS_PER_PROFILE="${SECONDS_PER_PROFILE:-30}"
JOBS="${JOBS:-8}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$INDICES:$SECONDS_PER_PROFILE:$JOBS" in
  *[!0-9,.:]*)
    echo "indices and solver settings contain invalid characters" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
PYTHON="${ROOT}/${WILDLIFE_VENV}/bin/python"
INPUT="${ROOT}/cascadiav3/fleet_inputs/${FLEET_TAG}/taskset.json"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
OUTPUT="${OUTPUT_DIR}/shard_${SHARD_HOST}.json"
LOG_DIR="${ROOT}/cascadiav3/logs"
HEARTBEAT="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.heartbeat"
EXIT_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.exit"
CHILD_PID_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.solver.pid"
WRAPPER_PID_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"
test -s "$INPUT"
test ! -e "$OUTPUT"
test ! -e "$EXIT_FILE"
test ! -e "$WRAPPER_PID_FILE"
[ "$(shasum -a 256 "$INPUT" | awk '{print $1}')" = "$TASKSET_SHA256" ]
[ "$(shasum -a 256 tools/all_wildlife_rules.py | awk '{print $1}')" = "$RULES_SOURCE_SHA256" ]
[ "$(shasum -a 256 tools/all_wildlife_exact.py | awk '{print $1}')" = "$EXACT_SOURCE_SHA256" ]
[ "$(shasum -a 256 tools/cbddb_wildlife_exact.py | awk '{print $1}')" = "$EXACT_SUPPORT_SHA256" ]
[ "$(shasum -a 256 tools/all_wildlife_profile_proof.py | awk '{print $1}')" = "$RUNNER_SOURCE_SHA256" ]
[ "$(shasum -a 256 "$0" | awk '{print $1}')" = "$WORKER_SHA256" ]
printf '%s\n' "$$" > "${WRAPPER_PID_FILE}.tmp"
mv "${WRAPPER_PID_FILE}.tmp" "$WRAPPER_PID_FILE"

printf '%s source=%s host=%s indices=%s jobs=%s seconds=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_REVISION" "$SHARD_HOST" \
  "$INDICES" "$JOBS" "$SECONDS_PER_PROFILE"
set +e
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u -m tools.all_wildlife_profile_proof \
  --taskset "$INPUT" --indices "$INDICES" --output "$OUTPUT" \
  --seconds "$SECONDS_PER_PROFILE" --jobs "$JOBS" &
solver_pid=$!
printf '%s\n' "$solver_pid" > "${CHILD_PID_FILE}.tmp"
mv "${CHILD_PID_FILE}.tmp" "$CHILD_PID_FILE"
while kill -0 "$solver_pid" 2>/dev/null; do
  printf '%s solver_pid=%s indices=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" "$INDICES" \
    > "${HEARTBEAT}.tmp"
  mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
  sleep 10
done
wait "$solver_pid"
status=$?
set -e

printf '%s complete indices=%s exit=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$INDICES" "$status" \
  > "${HEARTBEAT}.tmp"
mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
printf '%s\n' "$status" > "${EXIT_FILE}.tmp"
mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
exit "$status"
