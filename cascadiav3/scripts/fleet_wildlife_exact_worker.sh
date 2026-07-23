#!/usr/bin/env bash
set -euo pipefail

# Runs one exact wildlife-catalog shard on a Mac mini. This script is launched
# detached by fleet_wildlife_exact_launch.sh and owns a durable output ledger,
# heartbeat, and terminal exit-code file on the remote host.

RULESET="${RULESET:?set RULESET (aaaaa or cbddb)}"
FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX}"
SHARD_COUNT="${SHARD_COUNT:?set SHARD_COUNT}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
JOBS="${JOBS:-2}"
SOLVER_WORKERS="${SOLVER_WORKERS:-4}"
RELAXATION_TIME_LIMIT="${RELAXATION_TIME_LIMIT:-60}"
CONNECTED_TIME_LIMIT="${CONNECTED_TIME_LIMIT:-120}"
BASE_SEED="${BASE_SEED:-20260725}"
ORTOOLS_VERSION="${ORTOOLS_VERSION:?set ORTOOLS_VERSION}"
CATALOG_SOURCE_SHA256="${CATALOG_SOURCE_SHA256:?set CATALOG_SOURCE_SHA256}"
EXACT_SOURCE_SHA256="${EXACT_SOURCE_SHA256:?set EXACT_SOURCE_SHA256}"

case "$RULESET" in
  aaaaa)
    MODULE="tools.aaaaa_wildlife_catalog"
    CATALOG_SOURCE="tools/aaaaa_wildlife_catalog.py"
    EXACT_SOURCE="tools/aaaaa_wildlife_exact.py"
    ;;
  cbddb)
    MODULE="tools.cbddb_wildlife_catalog"
    CATALOG_SOURCE="tools/cbddb_wildlife_catalog.py"
    EXACT_SOURCE="tools/cbddb_wildlife_exact.py"
    ;;
  *)
    echo "unsupported RULESET=$RULESET" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
PYTHON="${ROOT}/venv/bin/python"
INPUT_DIR="${ROOT}/cascadiav3/fleet_inputs/${FLEET_TAG}"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"
OUTPUT="${OUTPUT_DIR}/shard_${SHARD_HOST}.json"
HEARTBEAT="${LOG_DIR}/wildlife_${FLEET_TAG}_shard_${SHARD_HOST}.heartbeat"
EXIT_FILE="${LOG_DIR}/wildlife_${FLEET_TAG}_shard_${SHARD_HOST}.exit"
CHILD_PID_FILE="${LOG_DIR}/wildlife_${FLEET_TAG}_shard_${SHARD_HOST}.solver.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"
test -s "$INPUT_DIR/candidates.json"
test -s "$INPUT_DIR/counts.json"
if [ -s "$EXIT_FILE" ] || [ -s "$OUTPUT" ]; then
  echo "existing terminal artifact for ${FLEET_TAG}/${SHARD_HOST}; refusing restart" >&2
  exit 65
fi

observed_ortools="$("$PYTHON" -c 'import ortools; print(ortools.__version__)')"
[ "$observed_ortools" = "$ORTOOLS_VERSION" ] || {
  echo "OR-Tools $observed_ortools, expected $ORTOOLS_VERSION" >&2
  exit 66
}
observed_catalog="$(shasum -a 256 "$CATALOG_SOURCE" | awk '{print $1}')"
observed_exact="$(shasum -a 256 "$EXACT_SOURCE" | awk '{print $1}')"
[ "$observed_catalog" = "$CATALOG_SOURCE_SHA256" ] || {
  echo "catalog source hash mismatch" >&2
  exit 67
}
[ "$observed_exact" = "$EXACT_SOURCE_SHA256" ] || {
  echo "exact source hash mismatch" >&2
  exit 68
}

args=(
  -u -m "$MODULE"
  --candidates "$INPUT_DIR/candidates.json"
  --counts-file "$INPUT_DIR/counts.json"
  --output "$OUTPUT"
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
  --jobs "$JOBS"
  --solver-workers "$SOLVER_WORKERS"
  --relaxation-time-limit "$RELAXATION_TIME_LIMIT"
  --connected-time-limit "$CONNECTED_TIME_LIMIT"
  --seed "$BASE_SEED"
)
if [ -s "$INPUT_DIR/import_ledger.json" ]; then
  args+=(--import-ledger "$INPUT_DIR/import_ledger.json")
fi

printf '%s source=%s ruleset=%s shard=%s/%s jobs=%s workers=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_REVISION" "$RULESET" \
  "$SHARD_INDEX" "$SHARD_COUNT" "$JOBS" "$SOLVER_WORKERS"

set +e
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "${args[@]}" &
solver_pid=$!
printf '%s\n' "$solver_pid" > "${CHILD_PID_FILE}.tmp"
mv "${CHILD_PID_FILE}.tmp" "$CHILD_PID_FILE"
while kill -0 "$solver_pid" 2>/dev/null; do
  printf '%s solver_pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" \
    > "${HEARTBEAT}.tmp"
  mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
  sleep 30
done
wait "$solver_pid"
status=$?
set -e

printf '%s solver_pid=%s exit=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" "$status" > "${HEARTBEAT}.tmp"
mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
printf '%s\n' "$status" > "${EXIT_FILE}.tmp"
mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
exit "$status"
