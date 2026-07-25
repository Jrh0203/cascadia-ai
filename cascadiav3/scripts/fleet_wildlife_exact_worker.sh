#!/usr/bin/env bash
set -euo pipefail

# Run one wildlife-catalog shard. Inputs are ordinary files and may be rerun
# or replaced. No source/version/hash checks and no receipt protocol.

RULESET="${RULESET:?set RULESET (aaaaa or cbddb)}"
FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:-$(hostname -s)}"
SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX}"
SHARD_COUNT="${SHARD_COUNT:?set SHARD_COUNT}"
JOBS="${JOBS:-2}"
SOLVER_WORKERS="${SOLVER_WORKERS:-4}"
RELAXATION_TIME_LIMIT="${RELAXATION_TIME_LIMIT:-60}"
CONNECTED_TIME_LIMIT="${CONNECTED_TIME_LIMIT:-120}"
BASE_SEED="${BASE_SEED:-1}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"
ROOT="${ROOT:-${HOME}/cascadia}"

case "$RULESET" in
  aaaaa) MODULE="tools.aaaaa_wildlife_catalog" ;;
  cbddb) MODULE="tools.cbddb_wildlife_catalog" ;;
  *) echo "unsupported RULESET=$RULESET" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-${ROOT}/${WILDLIFE_VENV}/bin/python}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/cascadiav3/fleet_inputs/${FLEET_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}}"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/shard_${SHARD_HOST}.json}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"
test -x "$PYTHON"
test -s "$INPUT_DIR/candidates.json"
test -s "$INPUT_DIR/counts.json"

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
if [ -s "$INPUT_DIR/import_catalog.json" ]; then
  args+=(--import-catalog "$INPUT_DIR/import_catalog.json")
fi

echo "[wildlife:$FLEET_TAG] host=$SHARD_HOST shard=$SHARD_INDEX/$SHARD_COUNT"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "${args[@]}"
