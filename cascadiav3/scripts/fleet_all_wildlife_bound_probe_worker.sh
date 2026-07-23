#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
TASK_INDICES="${TASK_INDICES:?set comma-separated TASK_INDICES}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
TASKSET_SHA256="${TASKSET_SHA256:?set TASKSET_SHA256}"
CATALOG_SHA256="${CATALOG_SHA256:?set CATALOG_SHA256}"
PROBE_SOURCE_SHA256="${PROBE_SOURCE_SHA256:?set PROBE_SOURCE_SHA256}"
EXACT_SOURCE_SHA256="${EXACT_SOURCE_SHA256:?set EXACT_SOURCE_SHA256}"
EXACT_SUPPORT_SHA256="${EXACT_SUPPORT_SHA256:?set EXACT_SUPPORT_SHA256}"
RULES_SOURCE_SHA256="${RULES_SOURCE_SHA256:?set RULES_SOURCE_SHA256}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"
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
if [ "$HEARTBEAT_INTERVAL" -lt 1 ] || [ "$HEARTBEAT_INTERVAL" -gt 60 ]; then
  echo "HEARTBEAT_INTERVAL must be an integer from 1 to 60" >&2
  exit 64
fi
case "$WILDLIFE_VENV" in
  ""|/*|*".."*|*[!A-Za-z0-9._/-]*)
    echo "WILDLIFE_VENV must be a safe relative path" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
PYTHON="${ROOT}/${WILDLIFE_VENV}/bin/python"
INPUT_DIR="${ROOT}/cascadiav3/fleet_inputs/${FLEET_TAG}"
TASKSET="${INPUT_DIR}/taskset.json"
CATALOG="${INPUT_DIR}/catalog.json"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"
HEARTBEAT="${LOG_DIR}/all_wildlife_bound_${FLEET_TAG}_${SHARD_HOST}.heartbeat"
EXIT_FILE="${LOG_DIR}/all_wildlife_bound_${FLEET_TAG}_${SHARD_HOST}.exit"
CHILD_PID_FILE="${LOG_DIR}/all_wildlife_bound_${FLEET_TAG}_${SHARD_HOST}.solver.pid"
WRAPPER_PID_FILE="${LOG_DIR}/all_wildlife_bound_${FLEET_TAG}_${SHARD_HOST}.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"
test -s "$TASKSET"
test -s "$CATALOG"
test ! -e "$EXIT_FILE"
test ! -e "$WRAPPER_PID_FILE"
test "$(shasum -a 256 "$TASKSET" | awk '{print $1}')" = "$TASKSET_SHA256"
test "$(shasum -a 256 "$CATALOG" | awk '{print $1}')" = "$CATALOG_SHA256"
test "$(shasum -a 256 tools/all_wildlife_bound_probe.py | awk '{print $1}')" = "$PROBE_SOURCE_SHA256"
test "$(shasum -a 256 tools/all_wildlife_exact.py | awk '{print $1}')" = "$EXACT_SOURCE_SHA256"
test "$(shasum -a 256 tools/cbddb_wildlife_exact.py | awk '{print $1}')" = "$EXACT_SUPPORT_SHA256"
test "$(shasum -a 256 tools/all_wildlife_rules.py | awk '{print $1}')" = "$RULES_SOURCE_SHA256"
printf '%s\n' "$$" > "${WRAPPER_PID_FILE}.tmp"
mv "${WRAPPER_PID_FILE}.tmp" "$WRAPPER_PID_FILE"

IFS=',' read -r -a task_array <<< "$TASK_INDICES"
for task_index in "${task_array[@]}"; do
  task="$("$PYTHON" - "$TASKSET" "$task_index" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
task_index = int(sys.argv[2])
if payload.get("schema") != "all-wildlife-bound-probe-taskset-v1":
    raise SystemExit("unexpected taskset schema")
tasks = payload.get("tasks", [])
if task_index < 0 or task_index >= len(tasks):
    raise SystemExit("task index out of range")
task = tasks[task_index]
if task.get("task_index") != task_index:
    raise SystemExit("task identity mismatch")
print(
    str(task["ruleset_index"])
    + ":"
    + ",".join(str(value) for value in task["counts"])
)
PY
)"
  ruleset_index="${task%%:*}"
  counts="${task#*:}"
  output="${OUTPUT_DIR}/task_${task_index}.json"
  printf '%s source=%s host=%s task=%s ruleset_index=%s counts=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_REVISION" "$SHARD_HOST" \
    "$task_index" "$ruleset_index" "$counts"
  set +e
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u -m tools.all_wildlife_bound_probe \
    --catalog "$CATALOG" --index "$ruleset_index" --counts "$counts" \
    --time-limit "$TIME_LIMIT" --total-time-limit "$TOTAL_TIME_LIMIT" \
    --workers "$SOLVER_WORKERS" --output "$output" --resume &
  solver_pid=$!
  printf '%s\n' "$solver_pid" > "${CHILD_PID_FILE}.tmp"
  mv "${CHILD_PID_FILE}.tmp" "$CHILD_PID_FILE"
  while kill -0 "$solver_pid" 2>/dev/null; do
    printf '%s solver_pid=%s task=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" "$task_index" \
      > "${HEARTBEAT}.tmp"
    mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
    sleep "$HEARTBEAT_INTERVAL"
  done
  wait "$solver_pid"
  status=$?
  set -e
  if [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then
    printf '%s\n' "$status" > "${EXIT_FILE}.tmp"
    mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
    exit "$status"
  fi
done

printf '%s complete tasks=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TASK_INDICES" \
  > "${HEARTBEAT}.tmp"
mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
printf '0\n' > "${EXIT_FILE}.tmp"
mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
