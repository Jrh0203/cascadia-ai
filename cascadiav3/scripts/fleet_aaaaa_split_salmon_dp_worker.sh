#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
CASE_INDEX="${CASE_INDEX:?set CASE_INDEX}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$CASE_INDEX" in
  0|1|2|3) ;;
  *)
    echo "CASE_INDEX must be 0, 1, 2, or 3" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
PYTHON="${ROOT}/${WILDLIFE_VENV}/bin/python"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
OUTPUT="${OUTPUT_DIR}/case_${CASE_INDEX}_${SHARD_HOST}.json"
LOG_DIR="${ROOT}/cascadiav3/logs"
PREFIX="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"

printf '%s host=%s case=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SHARD_HOST" "$CASE_INDEX"

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u -m tools.aaaaa_wildlife_split_salmon_dp_screen \
  --case-index "$CASE_INDEX" --output "$OUTPUT"
