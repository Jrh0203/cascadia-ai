#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
RANGE_START="${RANGE_START:?set RANGE_START}"
RANGE_END="${RANGE_END:?set RANGE_END}"
THREADS="${THREADS:-8}"
RESTARTS="${RESTARTS:-12}"
ITERATIONS="${ITERATIONS:-100000}"
BASE_SEED="${BASE_SEED:-20260723}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$RANGE_START:$RANGE_END:$THREADS:$RESTARTS:$ITERATIONS:$BASE_SEED" in
  *[!0-9:]*)
    echo "numeric worker arguments must be nonnegative integers" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
BINARY="${ROOT}/target/release/all_wildlife_candidates"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"
OUTPUT="${OUTPUT_DIR}/shard_${SHARD_HOST}.json"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$BINARY"

printf '%s host=%s range=[%s,%s) threads=%s restarts=%s iterations=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SHARD_HOST" \
  "$RANGE_START" "$RANGE_END" "$THREADS" "$RESTARTS" "$ITERATIONS"

"$BINARY" "$OUTPUT" "$RANGE_START" "$RANGE_END" "$THREADS" \
  "$RESTARTS" "$ITERATIONS" "$BASE_SEED"
