#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
INDICES="${INDICES:?set comma-separated INDICES}"
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

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$PYTHON"
test -s "$INPUT"

printf '%s host=%s indices=%s jobs=%s seconds=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SHARD_HOST" \
  "$INDICES" "$JOBS" "$SECONDS_PER_PROFILE"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u -m tools.all_wildlife_profile_proof \
  --taskset "$INPUT" --indices "$INDICES" --output "$OUTPUT" \
  --seconds "$SECONDS_PER_PROFILE" --jobs "$JOBS"
