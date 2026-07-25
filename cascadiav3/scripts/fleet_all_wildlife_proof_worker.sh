#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
INDICES="${INDICES:?set comma-separated INDICES}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"
TIME_LIMIT="${TIME_LIMIT:-30}"
TOTAL_TIME_LIMIT="${TOTAL_TIME_LIMIT:-300}"
SOLVER_WORKERS="${SOLVER_WORKERS:-4}"
CONNECTIVITY_REQUIRED="${CONNECTIVITY_REQUIRED:-1}"

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
case "$CONNECTIVITY_REQUIRED" in
  0|1) ;;
  *)
    echo "CONNECTIVITY_REQUIRED must be 0 or 1" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
PYTHON="${ROOT}/${WILDLIFE_VENV}/bin/python"
INPUT="${ROOT}/cascadiav3/fleet_inputs/${FLEET_TAG}/candidates.json"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"
test -s "$INPUT"

IFS=',' read -r -a index_array <<< "$INDICES"
proof_args=(
  --candidates "$INPUT"
  --time-limit "$TIME_LIMIT"
  --total-time-limit "$TOTAL_TIME_LIMIT"
  --workers "$SOLVER_WORKERS"
)
if [ "$CONNECTIVITY_REQUIRED" = 0 ]; then
  proof_args+=(--no-connectivity)
fi
for index in "${index_array[@]}"; do
  output="${OUTPUT_DIR}/ruleset_${index}.json"
  printf '%s host=%s index=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SHARD_HOST" "$index"
  set +e
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u -m tools.all_wildlife_global_proof \
    "${proof_args[@]}" --index "$index" --output "$output" --resume
  status=$?
  set -e
  if [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then
    exit "$status"
  fi
done
