#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
REMOTE_HOSTS="${REMOTE_HOSTS:-john2 john3 john4}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"

case "$FLEET_TAG" in
  *[!A-Za-z0-9._-]*)
    echo "FLEET_TAG must be a safe identifier" >&2
    exit 64
    ;;
esac
case "$SYNC_INTERVAL:$CHUNK_SIZE" in
  *[!0-9:]*)
    echo "SYNC_INTERVAL and CHUNK_SIZE must be nonnegative integers" >&2
    exit 64
    ;;
esac
if [ "$SYNC_INTERVAL" -lt 1 ] || [ "$CHUNK_SIZE" -lt 1 ]; then
  echo "SYNC_INTERVAL and CHUNK_SIZE must be positive" >&2
  exit 64
fi

ROOT="${HOME}/cascadia"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
SUMMARY="${OUTPUT_DIR}/summary.json"
PYTHON="${ROOT}/.venv/bin/python"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"
test -x "$PYTHON"

while true; do
  for host in $REMOTE_HOSTS; do
    if ! rsync -a --ignore-existing \
      --include='chunk_*.json' --exclude='*' \
      "${host}:cascadia/cascadiav3/fleet_outputs/${FLEET_TAG}/" \
      "${OUTPUT_DIR}/"; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) sync failed for ${host}" >&2
    fi
  done
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
    -m tools.all_wildlife_fixed_count_collect \
    --directories "$OUTPUT_DIR" --chunk-size "$CHUNK_SIZE" --output "$SUMMARY"
  complete="$("$PYTHON" - "$SUMMARY" <<'PY'
import json
import sys

print("1" if json.load(open(sys.argv[1]))["complete"] else "0")
PY
)"
  if [ "$complete" = "1" ]; then
    exit 0
  fi
  sleep "$SYNC_INTERVAL"
done
