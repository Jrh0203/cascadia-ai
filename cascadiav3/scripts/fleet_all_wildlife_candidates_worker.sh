#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
RANGE_START="${RANGE_START:?set RANGE_START}"
RANGE_END="${RANGE_END:?set RANGE_END}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
SOURCE_SHA256="${SOURCE_SHA256:?set SOURCE_SHA256}"
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
SOURCE="${ROOT}/crates/cascadia-game/src/bin/all_wildlife_candidates.rs"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"
OUTPUT="${OUTPUT_DIR}/shard_${SHARD_HOST}.json"
HEARTBEAT="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.heartbeat"
EXIT_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.exit"
CHILD_PID_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.solver.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$BINARY"
test ! -e "$OUTPUT"
test ! -e "$EXIT_FILE"
observed_sha="$(shasum -a 256 "$SOURCE" | awk '{print $1}')"
[ "$observed_sha" = "$SOURCE_SHA256" ] || {
  echo "candidate source hash mismatch: $observed_sha != $SOURCE_SHA256" >&2
  exit 65
}

printf '%s source=%s host=%s range=[%s,%s) threads=%s restarts=%s iterations=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_REVISION" "$SHARD_HOST" \
  "$RANGE_START" "$RANGE_END" "$THREADS" "$RESTARTS" "$ITERATIONS"

set +e
"$BINARY" "$OUTPUT" "$RANGE_START" "$RANGE_END" "$THREADS" \
  "$RESTARTS" "$ITERATIONS" "$BASE_SEED" &
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
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$solver_pid" "$status" \
  > "${HEARTBEAT}.tmp"
mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
printf '%s\n' "$status" > "${EXIT_FILE}.tmp"
mv "${EXIT_FILE}.tmp" "$EXIT_FILE"
exit "$status"
