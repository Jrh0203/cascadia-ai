#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
START_INDEX="${START_INDEX:?set START_INDEX}"
END_INDEX="${END_INDEX:?set END_INDEX}"
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
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"
OUTPUT="${OUTPUT_DIR}/shard_${SHARD_HOST}.json"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u -m tools.derive_hex_dual_observation_bounds \
  --start-index "$START_INDEX" --end-index "$END_INDEX" \
  --seconds "$SECONDS_PER_COMPONENT" --workers "$SOLVER_WORKERS" \
  --output "$OUTPUT"
