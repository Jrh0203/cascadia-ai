#!/usr/bin/env bash
set -euo pipefail

FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
INDICES="${INDICES:?set comma-separated INDICES}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
CANDIDATE_SOURCE_SHA256="${CANDIDATE_SOURCE_SHA256:?set CANDIDATE_SOURCE_SHA256}"
SUPPORT_SOURCE_SHA256="${SUPPORT_SOURCE_SHA256:?set SUPPORT_SOURCE_SHA256}"
BINARY_SHA256="${BINARY_SHA256:?set BINARY_SHA256}"
THREADS="${THREADS:-8}"
RESTARTS="${RESTARTS:-96}"
ITERATIONS="${ITERATIONS:-500000}"
BASE_SEED="${BASE_SEED:-2026072302}"

case "$FLEET_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "FLEET_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac
case "$INDICES:$THREADS:$RESTARTS:$ITERATIONS:$BASE_SEED" in
  *[!0-9,:]*)
    echo "indices and candidate settings contain invalid characters" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
BINARY="${ROOT}/target/release/all_wildlife_candidates"
CANDIDATE_SOURCE="${ROOT}/crates/cascadia-game/src/bin/all_wildlife_candidates.rs"
SUPPORT_SOURCE="${ROOT}/crates/cascadia-game/src/bin/wildlife_solver_support/mod.rs"
OUTPUT_DIR="${ROOT}/cascadiav3/fleet_outputs/${FLEET_TAG}"
LOG_DIR="${ROOT}/cascadiav3/logs"
HEARTBEAT="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.heartbeat"
EXIT_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.exit"
CHILD_PID_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.solver.pid"
WRAPPER_PID_FILE="${LOG_DIR}/all_wildlife_${FLEET_TAG}_${SHARD_HOST}.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -x "$BINARY"
test ! -e "$EXIT_FILE"
test ! -e "$WRAPPER_PID_FILE"

observed_candidate_source="$(shasum -a 256 "$CANDIDATE_SOURCE" | awk '{print $1}')"
observed_support_source="$(shasum -a 256 "$SUPPORT_SOURCE" | awk '{print $1}')"
observed_binary="$(shasum -a 256 "$BINARY" | awk '{print $1}')"
[ "$observed_candidate_source" = "$CANDIDATE_SOURCE_SHA256" ] || {
  echo "candidate source hash mismatch" >&2
  exit 65
}
[ "$observed_support_source" = "$SUPPORT_SOURCE_SHA256" ] || {
  echo "support source hash mismatch" >&2
  exit 65
}
[ "$observed_binary" = "$BINARY_SHA256" ] || {
  echo "candidate binary hash mismatch" >&2
  exit 65
}

IFS=',' read -r -a index_array <<< "$INDICES"
test "${#index_array[@]}" -gt 0
seen_indices=","
for index in "${index_array[@]}"; do
  [ "$index" -ge 0 ] && [ "$index" -lt 1024 ] || {
    echo "ruleset index out of range: $index" >&2
    exit 64
  }
  case "$seen_indices" in
    *",$index,"*)
      echo "duplicate ruleset index: $index" >&2
      exit 64
      ;;
  esac
  seen_indices="${seen_indices}${index},"
  test ! -e "${OUTPUT_DIR}/ruleset_${index}.json"
done

printf '%s\n' "$$" > "${WRAPPER_PID_FILE}.tmp"
mv "${WRAPPER_PID_FILE}.tmp" "$WRAPPER_PID_FILE"

for index in "${index_array[@]}"; do
  output="${OUTPUT_DIR}/ruleset_${index}.json"
  printf '%s source=%s host=%s index=%s threads=%s restarts=%s iterations=%s seed=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_REVISION" "$SHARD_HOST" \
    "$index" "$THREADS" "$RESTARTS" "$ITERATIONS" "$BASE_SEED"
  set +e
  "$BINARY" "$output" "$index" "$((index + 1))" "$THREADS" \
    "$RESTARTS" "$ITERATIONS" "$BASE_SEED" &
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
  if [ "$status" -ne 0 ]; then
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
