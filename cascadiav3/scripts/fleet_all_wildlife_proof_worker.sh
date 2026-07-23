#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
INDICES="${INDICES:?set comma-separated INDICES}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
CANDIDATE_SHA256="${CANDIDATE_SHA256:?set CANDIDATE_SHA256}"
PROOF_SOURCE_SHA256="${PROOF_SOURCE_SHA256:?set PROOF_SOURCE_SHA256}"
EXACT_SOURCE_SHA256="${EXACT_SOURCE_SHA256:?set EXACT_SOURCE_SHA256}"
EXACT_SUPPORT_SHA256="${EXACT_SUPPORT_SHA256:?set EXACT_SUPPORT_SHA256}"
RULES_SOURCE_SHA256="${RULES_SOURCE_SHA256:?set RULES_SOURCE_SHA256}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"
TIME_LIMIT="${TIME_LIMIT:-30}"
TOTAL_TIME_LIMIT="${TOTAL_TIME_LIMIT:-300}"
SOLVER_WORKERS="${SOLVER_WORKERS:-4}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$INDICES:$TIME_LIMIT:$TOTAL_TIME_LIMIT:$SOLVER_WORKERS" in
  *[!0-9,.:]*)
    echo "indices and solver settings contain invalid characters" >&2
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
INPUT="${ROOT}/cascadiav3/fleet_inputs/${FLEET_TAG}/candidates.json"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"
HEARTBEAT="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.heartbeat"
EXIT_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.exit"
CHILD_PID_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.solver.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"
test -s "$INPUT"
test ! -e "$EXIT_FILE"
observed_candidate="$(shasum -a 256 "$INPUT" | awk '{print $1}')"
observed_proof="$(shasum -a 256 tools/all_wildlife_global_proof.py | awk '{print $1}')"
observed_exact="$(shasum -a 256 tools/all_wildlife_exact.py | awk '{print $1}')"
observed_exact_support="$(shasum -a 256 tools/cbddb_wildlife_exact.py | awk '{print $1}')"
observed_rules="$(shasum -a 256 tools/all_wildlife_rules.py | awk '{print $1}')"
[ "$observed_candidate" = "$CANDIDATE_SHA256" ]
[ "$observed_proof" = "$PROOF_SOURCE_SHA256" ]
[ "$observed_exact" = "$EXACT_SOURCE_SHA256" ]
[ "$observed_exact_support" = "$EXACT_SUPPORT_SHA256" ]
[ "$observed_rules" = "$RULES_SOURCE_SHA256" ]

IFS=',' read -r -a index_array <<< "$INDICES"
for index in "${index_array[@]}"; do
  output="${OUTPUT_DIR}/ruleset_${index}.json"
  printf '%s source=%s host=%s index=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_REVISION" "$SHARD_HOST" "$index"
  set +e
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u -m tools.all_wildlife_global_proof \
    --candidates "$INPUT" --index "$index" --output "$output" \
    --time-limit "$TIME_LIMIT" --total-time-limit "$TOTAL_TIME_LIMIT" \
    --workers "$SOLVER_WORKERS" --resume &
  solver_pid=$!
  printf '%s\n' "$solver_pid" > "${CHILD_PID_FILE}.tmp"
  mv "${CHILD_PID_FILE}.tmp" "$CHILD_PID_FILE"
  while kill -0 "$solver_pid" 2>/dev/null; do
    printf '%s solver_pid=%s index=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" "$index" \
      > "${HEARTBEAT}.tmp"
    mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
    sleep 30
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

printf '%s complete indices=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$INDICES" \
  > "${HEARTBEAT}.tmp"
mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
printf '0\n' > "${EXIT_FILE}.tmp"
mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
